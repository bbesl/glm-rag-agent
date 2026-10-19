r"""
RAG 深水区第二课：混合检索——BM25 + 向量 + RRF 融合
=====================================================
上一课（第二十课）学了 Reranker：向量粗排 → LLM 精排。
但向量检索本身有个短板：精确关键词匹配弱。

举例：
  问"花呗怎么用" → 向量检索能返回"支付方式"相关 FAQ（语义近）
  但如果搜"100001"（订单号）→ 向量检索抓不到（数字没有语义）

BM25 关键词检索恰好相反：
  擅长精确关键词匹配（"花呗"两个字就在某条 FAQ 里 → 直接命中）
  但不擅长语义（"怎么退货"匹配不到"申请退货"——字面不同）

混合检索 = 两路互补：
  ① 向量检索：语义匹配（用 Chroma，复用之前课的逻辑）
  ② BM25 检索：关键词匹配（用 rank_bm25 库）
  ③ RRF 融合：把两路结果按排名融合，取最终 top-k

RRF（Reciprocal Rank Fusion，倒数排名融合）公式：
  score(d) = sum( 1 / (k + rank_i(d)) )
  - k 常取 60（常数，让排名靠后的也有一定权重，避免第一独大）
  - rank_i(d) = 文档 d 在第 i 路检索中的排名（从 1 开始）
  - 两路都命中的文档，分数叠加，自然排到前面

举例：文档 A 在向量检索排第 1，在 BM25 排第 3
  RRF 分 = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
  文档 B 只在向量检索排第 2：RRF 分 = 1/(60+2) = 0.0161
  → 文档 A（两路都靠前）> 文档 B（只一路靠前）

本课对比三种检索：向量 only / BM25 only / 混合 RRF

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\21_hybrid_search.py
"""
import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_embedding(text: str) -> list[float]:
    """调用智谱 embedding-3 生成向量（2048 维），复用第五课逻辑"""
    resp = client.embeddings.create(model="embedding-3", input=text)
    return resp.data[0].embedding


# ────────────────────────────────────────────
# 初始化 FAQ 知识库（复用第十三课 Chroma 持久化）
# ────────────────────────────────────────────
db_path = os.path.join(BASE_DIR, ".chroma_faq")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(
    name="faq_customer_service",
    metadata={"hnsw:space": "cosine"},
)

if collection.count() == 0:
    with open(os.path.join(BASE_DIR, "faq_knowledge.txt"), encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
    collection.add(
        documents=chunks,
        embeddings=[get_embedding(c) for c in chunks],
        ids=[f"faq_{i}" for i in range(len(chunks))],
    )
    print(f"已从 faq_knowledge.txt 读入 {len(chunks)} 条 FAQ")
else:
    print(f"FAQ 知识库就绪（{collection.count()} 条已入库）")

# 读入全部 FAQ 文档（BM25 需要原始文本建索引）
with open(os.path.join(BASE_DIR, "faq_knowledge.txt"), encoding="utf-8") as f:
    ALL_FAQ = [c.strip() for c in f.read().split("\n\n") if c.strip()]


# ════════════════════════════════════════════
# 第一路：向量语义检索（复用第十三课）
# ════════════════════════════════════════════
def vector_search(question: str, top_k: int = 5):
    """
    向量检索 top_k 条 FAQ。
    返回 [(rank, doc), ...] rank 从 1 开始（RRF 要用排名）。
    """
    q_vec = get_embedding(question)
    results = collection.query(query_embeddings=[q_vec], n_results=top_k)
    docs = results["documents"][0]
    return [(i + 1, doc) for i, doc in enumerate(docs)]


# ════════════════════════════════════════════
# 第二路：BM25 关键词检索（本课新增）
# ════════════════════════════════════════════
def tokenize(text: str) -> list[str]:
    """
    中文分词：字符级（每个中文字符一个 token），英文/数字按连续串保留。
    简单有效，无需装 jieba，适合教学。
    例：tokenize("花呗分期") → ["花", "呗", "分", "期"]
         tokenize("订单100001") → ["订", "单", "100001"]（"订单"拆成单字，数字保持连续）
    """
    tokens = []
    current = ""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            # 中文字符：先把攒着的英文/数字串收尾
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)  # 单字一个 token
        elif ch.isalnum():
            current += ch  # 英文/数字：攒成连续串
        else:
            if current:
                tokens.append(current)
                current = ""
    if current:
        tokens.append(current)
    return tokens


# 建一次 BM25 索引，全局复用（文档不变就不用重建）
BM25_INDEX = BM25Okapi([tokenize(doc) for doc in ALL_FAQ])


def bm25_search(question: str, top_k: int = 5):
    """
    BM25 检索 top_k 条 FAQ。
    返回 [(rank, doc), ...] rank 从 1 开始。

    BM25 原理（简化版 TF-IDF 进化）：
      score(d, q) = sum over terms t in q:  IDF(t) * TF(t,d) * (k1+1) / (TF(t,d) + k1*(1-b+b*|d|/avgdl))
      - TF = 词在文档出现次数（越多越好，但有饱和）
      - IDF = 词的稀有度（越稀有权重越高）
      - |d|/avgdl = 文档长度归一化（短文档里出现更值钱）
    rank_bm25 库帮我们算好了，只管调 get_scores
    """
    scores = BM25_INDEX.get_scores(tokenize(question))
    # 按分数降序取 top_k
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(i + 1, ALL_FAQ[idx]) for i, (idx, _) in enumerate(ranked)]


# ════════════════════════════════════════════
# 第三步：RRF 融合两路结果（本课核心）
# ════════════════════════════════════════════
def rrf_fuse(vector_results, bm25_results, k: int = 60, top_k: int = 3):
    """
    RRF（倒数排名融合）合并两路检索结果。

    公式：score(d) = sum( 1 / (k + rank(d)) )
    - k=60 常用常数（让排名靠后的也有权重，避免第 1 名独大）
    - rank(d) = 文档 d 在某路检索中的排名（从 1 开始）
    - 两路都命中的文档分数叠加，自然排到前面

    输入：两路 [(rank, doc), ...]
    输出：[(score, doc), ...] 按 RRF 分数降序
    """
    scores = {}
    for rank, doc in vector_results:
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    for rank, doc in bm25_results:
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    # 按分数降序
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc) for doc, score in ranked[:top_k]]


# ════════════════════════════════════════════
# 对比演示：三种检索方式
# ════════════════════════════════════════════
def show_results(title, results):
    """打印检索结果，每条只显示 FAQ 标题（—— 前的部分）方便对比"""
    print(f"\n  {title}")
    print(f"  {'-'*50}")
    for i, (score_or_rank, doc) in enumerate(results):
        label = doc.split("——")[0] if "——" in doc else doc[:18]
        print(f"  {i+1}. [{score_or_rank:>7.4f}] {label}")
    return [doc for _, doc in results]


# 测试问题设计：三种典型场景
TEST_QUESTIONS = [
    ("怎么退货", "语义问题——向量赢（'怎么退货'→匹配'申请退货'，字面不同但语义近）"),
    ("花呗", "关键词问题——BM25 赢（精确匹配'花呗'两字）"),
    ("新疆运费多少", "混合问题——两路都有信号，RRF 综合最优"),
]

if __name__ == "__main__":
    for question, note in TEST_QUESTIONS:
        print(f"\n{'#'*60}")
        print(f"  问题：{question}")
        print(f"  说明：{note}")
        print(f"{'#'*60}")

        vec = vector_search(question, top_k=5)
        bm = bm25_search(question, top_k=5)
        fused = rrf_fuse(vec, bm, k=60, top_k=3)

        vec_docs = show_results("向量检索 top-5（语义匹配）", vec)
        bm_docs = show_results("BM25 检索 top-5（关键词匹配）", bm)
        fused_docs = show_results("RRF 融合 top-3（混合）", fused)

        # 分析两路交集（RRF 加强的就是这部分）
        overlap = set(vec_docs[:3]) & set(bm_docs[:3])
        print(f"\n  两路 top-3 交集（RRF 加强的）：{len(overlap)} 条")
        for doc in overlap:
            label = doc.split("——")[0] if "——" in doc else doc[:18]
            print(f"    → {label}")

    print(f"\n{'='*60}")
    print("三种检索对比小结：")
    print("  向量 = 语义近但关键词可能不命中（'怎么退货'→'申请退货'）")
    print("  BM25 = 关键词精确命中但语义弱（'花呗'直接匹配）")
    print("  RRF  = 两路综合，两路都靠前的文档排最前（最稳）")
    print(f"{'='*60}")
