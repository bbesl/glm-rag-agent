r"""
第十四课：智能客服 Agent（完整版）
第十三课的客服只会"查 FAQ"——每个问题都走固定的"检索+生成"一条路。
这一课升级为完整 Agent：模型自己决定用哪个工具、调几次、什么时候停。

三个工具：
  1. search_faq(question)   —— 检索 FAQ 知识库（复用第十三课的 RAG）
  2. check_order(order_id)  —— 查订单状态（mock 数据模拟查数据库）
  3. transfer_human(reason) —— 转人工客服（标记交接原因）

本课结构 = 第十课 Agent 循环 + 第十一课 多轮记忆 + 第十三课 客服场景
  外层 while True + input()   —— 多轮对话（第十一课思路）
  内层 run_agent for 循环     —— 单轮内多步工具调用（第十课思路）
  messages 跨轮累积不重置      —— 这就是"记忆"（第十一课核心）

和第十三课的区别：
  - 第十三课：每个问题都强制走"检索FAQ→生成"，只会答FAQ
  - 第十四课：模型判断该查FAQ、查订单、还是转人工，能办更多事
  - 多了对话记忆：用户说"那它呢"，能接住上文的订单号

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\13_customer_service_agent.py
"""
import os
import json
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
    """调用智谱 embedding-3 生成向量（2048 维），复用第五课逻辑"""
    resp = client.embeddings.create(model="embedding-3", input=text)
    return resp.data[0].embedding


# ────────────────────────────────────────────
# 第一步：初始化 FAQ 知识库（Chroma 持久化，复用第十三课）
# 库已在第十三课建好，这里 count()>0 会直接复用，不重算向量
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


# ────────────────────────────────────────────
# 第二步：定义 3 个工具函数
# 真实项目里这些会连数据库/订单服务/工单系统，这里用 mock 演示
# ────────────────────────────────────────────

# 工具1：search_faq —— RAG 检索 FAQ（复用第十三课 retrieve 逻辑）
def search_faq(question: str) -> dict:
    """检索 FAQ 知识库，返回最相关的 3 条"""
    q_vec = get_embedding(question)
    results = collection.query(query_embeddings=[q_vec], n_results=3)
    # Chroma 返回嵌套列表，单次查询取 [0]（第八课踩过的坑）
    docs = results["documents"][0]
    dists = results["distances"][0]
    faqs = [
        {"content": d, "distance": round(dist, 3)}
        for d, dist in zip(docs, dists)
    ]
    return {"count": len(faqs), "faqs": faqs}


# 工具2：check_order —— 查订单状态（mock 数据）
# 真实项目里这里会 SELECT * FROM orders WHERE id=?
ORDERS = {
    "100001": {"status": "已发货", "carrier": "顺丰速运", "tracking": "SF1234567890", "eta": "预计明天送达"},
    "100002": {"status": "待发货", "carrier": None, "tracking": None, "eta": "预计今天内发出"},
    "100003": {"status": "已签收", "carrier": "圆通速递", "tracking": "YT9876543210", "eta": "已于昨天签收"},
    "100004": {"status": "已退款", "carrier": None, "tracking": None, "eta": "退款已到账"},
}


def check_order(order_id: str) -> dict:
    """按订单号查订单状态"""
    if order_id in ORDERS:
        return {"order_id": order_id, **ORDERS[order_id]}
    return {"error": f"未找到订单 {order_id}，请确认订单号是否正确"}


# 工具3：transfer_human —— 转人工客服
def transfer_human(reason: str) -> dict:
    """标记需要转人工，返回交接信息"""
    return {
        "transfer": True,
        "reason": reason,
        "message": "已为您接入人工客服，排队中预计等待1分钟",
    }


# ────────────────────────────────────────────
# 第三步：把 3 个工具用 JSON Schema 声明给模型（Function Calling）
# ────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "检索客服FAQ知识库。当用户问发货时间、退货流程、支付方式、保修期、发票等常见问题时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "用户的问题，用于语义检索FAQ"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order",
            "description": "查询订单状态。当用户提供订单号问发货进度、物流、是否签收时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，如 '100001'"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_human",
            "description": "转人工客服。当用户明确要求人工、要投诉、或问题超出FAQ与订单范围时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "转人工的原因"},
                },
                "required": ["reason"],
            },
        },
    },
]

# 工具名 → Python 函数 的映射，方便循环里统一调用（复用第十课写法）
tool_map = {
    "search_faq": search_faq,
    "check_order": check_order,
    "transfer_human": transfer_human,
}


# ────────────────────────────────────────────
# 第四步：Agent 主循环（复用第十一课 run_agent 模式）
# 和第十一课几乎一样，区别只是 tools 换成了客服三件套
# ────────────────────────────────────────────
def run_agent(messages: list, max_turns: int = 10) -> tuple:
    """
    在已有的 messages 历史上跑一轮 Agent。
    messages 会被【原地扩展】：追加模型回复和工具结果。
    返回 (最终回答文本, 本轮内循环了几轮)。
    """
    for turn in range(1, max_turns + 1):
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,   # 带着完整历史发给模型——这就是"记忆"
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # 模型给最终答案了
            # 关键：把模型的回答也追加进历史，下一轮对话它才"记得自己说过什么"
            messages.append(msg)
            return msg.content, turn

        # 要调工具：把模型回复追加，执行工具，喂回结果
        messages.append(msg)
        print(f"   [内第{turn}轮] 调用工具：", end="")
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            result = tool_map[name](**args)
            print(f" {name}({args})→{result}", end="")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        print()
        # 喂回后回到循环开头，模型继续判断要不要再调

    return "（达到最大轮次）", max_turns


# ────────────────────────────────────────────
# 第五步：交互式客服 Agent 主循环（复用第十一课双层循环）
# ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("第十四课：智能客服 Agent（完整版）")
    print("能查FAQ · 能查订单 · 能转人工 · 还能多轮记忆上下文")
    print("试试问：发货多久到？ / 订单100001到哪了？ / 我要投诉转人工")
    print("输入 quit / 退出 / q 结束对话")
    print("=" * 60)

    # 记忆的核心：messages 在 while 之外初始化，跨轮累积不重置
    # system 必须强调"调工具禁止编造"——glm-4-flash 不加会自己编订单号
    messages = [
        {
            "role": "system",
            "content": (
                "你是「好物优选」电商平台的智能客服。你有三个工具："
                "search_faq（查FAQ知识库）、check_order（查订单状态）、transfer_human（转人工）。\n"
                "规则：\n"
                "1) 回答事实性问题（FAQ、订单）必须调用对应工具拿真实数据，禁止编造订单号、物流单号、政策；\n"
                "2) FAQ能答的用search_faq；涉及订单进度用check_order；用户要人工/投诉/超出范围用transfer_human；\n"
                "3) 语气专业、简洁、有温度；\n"
                "4) 注意对话上下文，用户说'那个订单呢'指上句提到的订单号；\n"
                "5) 简单问候寒暄不必调工具。"
            ),
        }
    ]

    while True:
        try:
            user_input = input("\n客户：").strip()
        except EOFError:
            # 管道喂入测试时，输入读完会触发 EOFError，正常退出
            print("\n（输入结束，退出）")
            break

        if user_input.lower() in ("quit", "退出", "q", "exit"):
            print("客服：感谢咨询，祝您购物愉快！")
            break
        if not user_input:
            continue

        # 把用户这一句追加进历史
        messages.append({"role": "user", "content": user_input})

        print("客服正在处理...", end="\r")
        answer, turns = run_agent(messages)

        print(f"客服：{answer}")
        print(f"（本轮处理 {turns} 轮）")


if __name__ == "__main__":
    main()
