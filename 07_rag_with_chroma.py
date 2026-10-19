r"""
第八课：实战 RAG——向量数据库 Chroma + 真实文件
前几课的向量都临时算好放内存，文档还硬编码在代码里。这课做两个实战升级：
① 用 Chroma 向量库把向量持久化到磁盘（下次运行直接复用，不重算）
② 从真实 txt 文件读文档（以后换 PDF/Word 只改读取那行）

Chroma 是轻量级向量数据库，纯 Python、嵌入式（像 SQLite），不用单独起服务。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\07_rag_with_chroma.py
第一次运行会建库写入；再运行一次会发现"库已有数据，直接复用"——这就是持久化。
"""
import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# 用 __file__ 定位本脚本所在目录，避免终端 cwd 不对导致找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_embedding(text: str) -> list[float]:
    """复用第五课的 embedding 调用"""
    resp = client.embeddings.create(model="embedding-3", input=text)
    return resp.data[0].embedding


# ────────────────────────────────────────────
# 第一步：从真实文件读文档，按段落切分
# ────────────────────────────────────────────
doc_path = os.path.join(BASE_DIR, "knowledge.txt")
with open(doc_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 按空行分段——每段是一个 chunk
chunks = [c.strip() for c in raw_text.split("\n\n") if c.strip()]

print("=" * 56)
print("第一步：从 knowledge.txt 读文档")
print("=" * 56)
print(f"文件：{doc_path}")
print(f"切分成 {len(chunks)} 块，示例：{chunks[0][:30]}...")

# ────────────────────────────────────────────
# 第二步：初始化 Chroma 向量库（持久化到磁盘）
# ────────────────────────────────────────────
db_path = os.path.join(BASE_DIR, ".chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)

# metadata 里指定 cosine 距离——和第五课的余弦相似度一致
# 注意：Chroma 返回的是"距离"（越小越像），第五课是"相似度"（越大越像），方向相反
collection = chroma_client.get_or_create_collection(
    name="xyz_company",
    metadata={"hnsw:space": "cosine"},
)

# ────────────────────────────────────────────
# 第三步：入库——只在库为空时写入，已存在则复用（持久化的价值）
# ────────────────────────────────────────────
print()
print("=" * 56)
print("第二步：向量入库")
print("=" * 56)
if collection.count() == 0:
    print(f"库为空，正在把 {len(chunks)} 块文档向量化并写入 Chroma...")
    embeddings = [get_embedding(c) for c in chunks]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    print("写入完成！下次运行就不用重算向量了。")
else:
    print(f"库已有 {collection.count()} 条数据，直接复用（向量已存在磁盘）")


# ────────────────────────────────────────────
# 第四步：检索 + 生成（和第七课一样的 RAG 流程，但检索交给 Chroma）
# ────────────────────────────────────────────
def retrieve(question: str, top_k: int = 2):
    """用 Chroma 检索：传问题的向量，返回最相关的 top_k 块"""
    q_vec = get_embedding(question)
    results = collection.query(
        query_embeddings=[q_vec],   # 传我们算好的向量
        n_results=top_k,
    )
    # results 是嵌套结构：documents=[[块1,块2]], distances=[[d1,d2]]
    docs = results["documents"][0]
    dists = results["distances"][0]
    return list(zip(dists, docs))


def ask(question: str, top_k: int = 2) -> tuple[str, list]:
    """完整 RAG：检索 → 拼 prompt → 模型回答"""
    top = retrieve(question, top_k)
    context = "\n".join(f"片段{i+1}：{doc}" for i, (_, doc) in enumerate(top))
    messages = [
        {"role": "system",
         "content": f"你是文档助手。只能根据下面文档片段回答，"
                    f"文档里没有的就说不知道，禁止编造。\n\n{context}"},
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(model="glm-4-flash", messages=messages)
    return resp.choices[0].message.content, top


# ────────────────────────────────────────────
# 第五步：测试三个问题
# ────────────────────────────────────────────
QUESTIONS = [
    "报销超过 20000 元谁审批？",      # 应命中报销流程块
    "请假超过 3 天谁审批？",          # 应命中请假流程块
    "入职当天要做什么？",             # 应命中入职流程块
]

for q in QUESTIONS:
    print()
    print("=" * 56)
    print(f"问题：{q}")
    print("=" * 56)
    answer, top = ask(q)
    print("检索到的文档块（距离越小越相似）：")
    for dist, doc in top:
        print(f"  - (距离 {dist:.3f}) {doc[:35]}...")
    print("\n回答：" + answer)

print()
print("=" * 56)
print("完整实战 RAG！再跑一次本脚本，会看到\"直接复用\"——持久化生效。")
print("=" * 56)
