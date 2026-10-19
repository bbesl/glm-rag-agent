r"""
第十五课：智能客服 Agent —— Streamlit Web 版（最后一课）
第十四课的客服在命令行里跑——用户得开终端、敲 python 命令。
这课把它变成浏览器里能用的网页应用：打开链接就能聊。

用 Streamlit：一个 Python 库，写一个 .py 文件就能起一个 Web 服务。
核心替换（命令行 → Web）：
  input()            → st.chat_input()           浏览器输入框
  print()            → st.chat_message().write()  气泡
  messages 列表      → st.session_state.messages  跨交互不重置（记忆）
  工具调用过程       → st.status 折叠面板         可见 Agent 思考链路

Agent 内核（run_agent + 3 工具 + RAG 检索）原样复用第十四课，只换"外壳"。
这就是分层的价值：业务逻辑和界面解耦，换个 UI 不动内核。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe -m streamlit run ai-learning\15_customer_service_web.py
浏览器会自动打开 http://localhost:8501
"""
import os
import json
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

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
# 第一步：初始化 FAQ 知识库（Chroma 持久化，复用第十三/十四课）
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


# ────────────────────────────────────────────
# 第二步：3 个工具函数（复用第十四课，内核不动）
# ────────────────────────────────────────────
def search_faq(question: str) -> dict:
    """检索 FAQ 知识库，返回最相关的 3 条"""
    q_vec = get_embedding(question)
    results = collection.query(query_embeddings=[q_vec], n_results=3)
    docs = results["documents"][0]
    dists = results["distances"][0]
    faqs = [{"content": d, "distance": round(x, 3)} for d, x in zip(docs, dists)]
    return {"count": len(faqs), "faqs": faqs}


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


def transfer_human(reason: str) -> dict:
    """标记需要转人工，返回交接信息"""
    return {"transfer": True, "reason": reason, "message": "已为您接入人工客服，排队中预计等待1分钟"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "检索客服FAQ知识库，回答通用政策类问题。当用户问发货时间/发货多久/几天到、退货流程、支付方式、保修期、发票等【不需要订单号就能回答的通用问题】时调用。",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "用户的问题，用于语义检索FAQ"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order",
            "description": "查询某个具体订单的状态。只有当用户提供了明确的订单号（如100001）问该订单的发货进度、物流、是否签收时才调用。用户没给订单号时不要调用本工具。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号，如 '100001'"}},
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
                "properties": {"reason": {"type": "string", "description": "转人工的原因"}},
                "required": ["reason"],
            },
        },
    },
]

tool_map = {"search_faq": search_faq, "check_order": check_order, "transfer_human": transfer_human}


# ────────────────────────────────────────────
# 第三步：Agent 循环（复用第十四课，只多返回一个工具调用日志用于 Web 展示）
# ────────────────────────────────────────────
def run_agent(messages: list, max_turns: int = 10) -> tuple:
    """
    在已有 messages 历史上跑一轮 Agent。
    返回 (最终回答文本, 工具调用日志列表)。
    工具日志给 Web 界面的 st.status 展示 Agent 思考链路。
    """
    tool_log = []
    for turn in range(1, max_turns + 1):
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # 模型给最终答案了——存成 dict 方便 Web 渲染
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, tool_log

        # 要调工具：把模型回复追加，执行工具，喂回结果
        messages.append(msg)
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            result = tool_map[name](**args)
            tool_log.append({"tool": name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        # 喂回后回到循环开头，模型继续判断

    return "（达到最大轮次）", tool_log


# ────────────────────────────────────────────
# 第四步：Streamlit Web 界面
# ────────────────────────────────────────────
st.set_page_config(page_title="好物优选智能客服", page_icon="🛍️", layout="centered")

st.title("🛍️ 好物优选 · 智能客服")
st.caption("能查 FAQ · 能查订单 · 能转人工 · 还能多轮记住上下文")
st.divider()

SYSTEM_PROMPT = (
    "你是「好物优选」电商平台的智能客服。你有三个工具："
    "search_faq（查FAQ知识库）、check_order（查订单状态）、transfer_human（转人工）。\n"
    "规则：\n"
    "1) 回答事实性问题（FAQ、订单）必须调用对应工具拿真实数据，禁止编造订单号、物流单号、政策；\n"
    "2) FAQ能答的用search_faq；涉及订单进度用check_order；用户要人工/投诉/超出范围用transfer_human；\n"
    "3) 语气专业、简洁、有温度；\n"
    "4) 注意对话上下文，用户说'那个订单呢'指上句提到的订单号；\n"
    "5) 简单问候寒暄不必调工具。"
)

# 记忆核心：session_state 跨交互不重置（对应命令行的 messages 提到 while 外）
# 用户每次发消息触发脚本重跑，但 session_state 保留——这就是 Web 版的"记忆"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


# 渲染历史对话（跳过 system、tool、以及带 tool_calls 的空 content 消息）
def render_history():
    for m in st.session_state.messages:
        # messages 里混存 dict 和 OpenAI 对象，统一取 role/content
        if isinstance(m, dict):
            role, content = m["role"], m["content"]
        else:
            role, content = m.role, m.content
        if role in ("system", "tool"):
            continue
        if not content:
            # 带 tool_calls 的 assistant 消息 content 为 None，不渲染成气泡
            continue
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)


render_history()

# 输入框（浏览器底部固定输入栏）
if prompt := st.chat_input("问问客服吧，比如：发货多久到？ / 订单100001到哪了？"):
    # 渲染用户气泡
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 渲染客服气泡 + 折叠展示工具调用链路
    with st.chat_message("assistant"):
        with st.status("客服正在处理...", expanded=False) as status:
            answer, tool_log = run_agent(st.session_state.messages)
            # 展示 Agent 这一轮调了哪些工具（思考链路可见）
            for log in tool_log:
                st.write(f"🔧 调用 {log['tool']}({log['args']})")
                st.write(f"   → {log['result']}")
            status.update(label=f"处理完成（调了 {len(tool_log)} 次工具）", state="complete")
        st.markdown(answer)
