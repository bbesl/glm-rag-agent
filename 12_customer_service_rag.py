r"""
第十三课：实战项目——智能客服 Agent（第一部分：基础 RAG 客服）
从这课开始进入第五阶段"实战项目"，目标是把前面学的 RAG + Agent + 记忆 + 多 Agent
组装成一个完整的、能写进简历的 AI 应用。

这课先搭"地基"：一个电商客服 FAQ 问答系统。
用户提问 → 检索 FAQ 知识库 → 模型根据检索到的内容回答。
其实就是第八课（Chroma 持久化）+ 第九课（交互循环）的组合，
但换了真实业务场景（电商客服），为第十四课升级成完整 Agent 做准备。

和第八~九课的区别：
- 知识库从虚构的 XYZ 公司 → 真实的电商 FAQ（物流/退换/支付/账户/售后）
- system prompt 从"文档助手" → "客服角色"（有温度、会引导、答不了会建议转人工）
- 检索 top_k 从 2 → 3（FAQ 条目短，多取一条更准）

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\12_customer_service_rag.py
然后输入客服问题试试，输入 quit 退出。
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

# 用 __file__ 定位脚本目录，避免终端 cwd 不对导致找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_embedding(text: str) -> list[float]:
    """调用智谱 embedding-3 生成向量（2048 维），复用第五课逻辑"""
    resp = client.embeddings.create(model="embedding-3", input=text)
    return resp.data[0].embedding


# ────────────────────────────────────────────
# 第一步：初始化 FAQ 知识库（Chroma 持久化）
# ────────────────────────────────────────────
# 用新的库名和路径，和第八课的 xyz_company 库分开
db_path = os.path.join(BASE_DIR, ".chroma_faq")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(
    name="faq_customer_service",
    metadata={"hnsw:space": "cosine"},
)

# 库为空时从 faq_knowledge.txt 读入（和第八课一样的"只入库一次"逻辑）
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


# ────────────────────────────────────────────
# 第二步：检索 + 生成（RAG 核心流程）
# ────────────────────────────────────────────
def retrieve(question: str, top_k: int = 3):
    """用 Chroma 检索 top_k 条最相关的 FAQ"""
    q_vec = get_embedding(question)
    results = collection.query(query_embeddings=[q_vec], n_results=top_k)
    # Chroma 返回嵌套列表，单次查询取 [0]
    docs = results["documents"][0]
    dists = results["distances"][0]
    return list(zip(dists, docs))


def ask(question: str, top_k: int = 3) -> str:
    """完整 RAG：检索 FAQ → 拼进 system prompt → 模型回答"""
    top = retrieve(question, top_k)
    # 把检索到的 FAQ 拼成 context
    context = "\n".join(f"FAQ{i+1}：{doc}" for i, (_, doc) in enumerate(top))
    messages = [
        {
            "role": "system",
            "content": (
                "你是「好物优选」电商平台的智能客服。请根据下面的 FAQ 知识库回答客户问题。"
                "要求：1) 只用知识库里的信息，不要编造；"
                "2) 语气专业、简洁、有温度；"
                "3) 如果知识库里没有相关信息，礼貌告知并建议联系人工客服。\n\n"
                f"{context}"
            ),
        },
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(model="glm-4-flash", messages=messages)
    return resp.choices[0].message.content


# ────────────────────────────────────────────
# 第三步：交互式客服问答（复用第九课的循环）
# ────────────────────────────────────────────
print()
print("=" * 56)
print("「好物优选」智能客服（输入问题咨询，输入 quit 退出）")
print("=" * 56)

while True:
    try:
        q = input("\n客户：").strip()
    except EOFError:
        print("\n（输入结束）")
        break

    if not q:
        continue

    if q.lower() in ("quit", "exit", "退出", "q"):
        print("客服：感谢咨询，祝您购物愉快！")
        break

    print("客服正在查询...\r", end="")
    answer = ask(q)
    print(f"客服：{answer}")

print()
print("=" * 56)
print("这是基础版客服——只会查 FAQ 回答。")
print("下一课升级为完整 Agent：能查订单、能转人工、能多轮记忆！")
print("=" * 56)
