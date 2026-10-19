r"""
第九课：交互式问答程序——把 RAG 变成能连续对话的程序
前八课的脚本都是"写死问题、跑一次就退出"。这一课加个 while 循环：
你在终端输入问题 → 程序检索+回答 → 等你下一个问题 → 输入 quit 退出。

这就是真实 AI 应用的雏形：有循环、有交互、有状态（对话历史）。
下一阶段 Agent 就是在这个基础上加"工具调用"和"记忆"。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\08_interactive_rag.py
然后输入问题，输入 quit 退出。
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_embedding(text: str) -> list[float]:
    """复用第五课的 embedding 调用"""
    resp = client.embeddings.create(model="embedding-3", input=text)
    return resp.data[0].embedding


# ────────────────────────────────────────────
# 初始化 Chroma 向量库（复用第八课的逻辑）
# ────────────────────────────────────────────
db_path = os.path.join(BASE_DIR, ".chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(
    name="xyz_company",
    metadata={"hnsw:space": "cosine"},
)

# 库为空时从 knowledge.txt 读入
if collection.count() == 0:
    with open(os.path.join(BASE_DIR, "knowledge.txt"), encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
    collection.add(
        documents=chunks,
        embeddings=[get_embedding(c) for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    print(f"已从 knowledge.txt 读入 {len(chunks)} 块文档")
else:
    print(f"向量库就绪（{collection.count()} 块文档已入库）")


# ────────────────────────────────────────────
# 检索 + 生成（复用第八课的函数）
# ────────────────────────────────────────────
def retrieve(question: str, top_k: int = 2):
    q_vec = get_embedding(question)
    results = collection.query(query_embeddings=[q_vec], n_results=top_k)
    docs = results["documents"][0]
    dists = results["distances"][0]
    return list(zip(dists, docs))


def ask(question: str, top_k: int = 2) -> str:
    """完整 RAG：检索 → 拼 prompt → 模型回答"""
    top = retrieve(question, top_k)
    context = "\n".join(f"片段{i+1}：{doc}" for i, (_, doc) in enumerate(top))
    messages = [
        {"role": "system",
         "content": f"你是 XYZ 公司文档助手。只能根据下面文档片段回答，"
                    f"文档里没有的就说\"我不知道\"，禁止编造。\n\n{context}"},
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(model="glm-4-flash", messages=messages)
    return resp.choices[0].message.content


# ────────────────────────────────────────────
# 交互循环：这一课的新东西
# ────────────────────────────────────────────
print()
print("=" * 56)
print("XYZ 公司知识问答系统（输入 quit 退出）")
print("=" * 56)

while True:
    try:
        q = input("\n你的问题：").strip()
    except EOFError:
        # 管道输入结束（非交互模式）时也会走到这里
        print("\n（输入结束）")
        break

    # 空输入跳过
    if not q:
        continue

    # 退出命令
    if q.lower() in ("quit", "exit", "退出", "q"):
        print("再见！")
        break

    # 正常问答
    print("思考中...\r", end="")
    answer = ask(q)
    print(f"回答：{answer}")
