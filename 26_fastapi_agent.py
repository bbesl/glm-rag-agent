r"""
第二十六课：FastAPI——把客服 Agent 升级为生产级 API 服务
==========================================================
第十五课的 Streamlit 版是"演示级"Web（单人本地玩）；真实业务的后端
都是 HTTP API：前端 App/小程序/别的系统来调你的接口。本课把同一个
Agent 内核包装成标准 REST API。

【本课最重要的架构问题】
HTTP 协议是无状态的——每个请求互相独立，服务器记不住"你是谁"。
但多轮对话必须有记忆。解决方案（本课核心）：
    客户端每次带 session_id → 服务端用字典存每个 session 的 messages
    → 同一 session_id 的请求自动接上同一份对话历史
    （生产环境字典换 Redis，道理一模一样）

三层替换（命令行→Streamlit→API，内核始终不动）：
  input()              → st.chat_input()   → POST /chat 请求体
  print()              → st.chat_message() → JSON 响应
  while 外的 messages  → st.session_state  → SESSIONS[session_id]

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\26_fastapi_agent.py
启动后：
    交互文档  http://127.0.0.1:8000/docs      （浏览器打开，自动生成！）
    健康检查  http://127.0.0.1:8000/health
"""
import os
import json
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field
import uvicorn

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ────────────────────────────────────────────
# 第一步：FAQ 知识库（复用第十三~十五课，原样）
# ────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(model="embedding-3", input=text)
    return resp.data[0].embedding


chroma_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, ".chroma_faq"))
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
# 第二步：3 个工具 + tools 声明（复用第十四/十五课，原样）
# ────────────────────────────────────────────
def search_faq(question: str) -> dict:
    q_vec = get_embedding(question)
    results = collection.query(query_embeddings=[q_vec], n_results=3)
    faqs = [
        {"content": d, "distance": round(x, 3)}
        for d, x in zip(results["documents"][0], results["distances"][0])
    ]
    return {"count": len(faqs), "faqs": faqs}


ORDERS = {
    "100001": {"status": "已发货", "carrier": "顺丰速运", "tracking": "SF1234567890", "eta": "预计明天送达"},
    "100002": {"status": "待发货", "carrier": None, "tracking": None, "eta": "预计今天内发出"},
    "100003": {"status": "已签收", "carrier": "圆通速递", "tracking": "YT9876543210", "eta": "已于昨天签收"},
    "100004": {"status": "已退款", "carrier": None, "tracking": None, "eta": "退款已到账"},
}


def check_order(order_id: str) -> dict:
    if order_id in ORDERS:
        return {"order_id": order_id, **ORDERS[order_id]}
    return {"error": f"未找到订单 {order_id}，请确认订单号是否正确"}


def transfer_human(reason: str) -> dict:
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


# ────────────────────────────────────────────
# 第三步：Agent 循环（复用第十五课，原样——三层外壳内核不动）
# ────────────────────────────────────────────
def run_agent(messages: list, max_turns: int = 10) -> tuple:
    tool_log = []
    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model="glm-4-flash", messages=messages, tools=tools, tool_choice="auto"
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, tool_log
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
    return "（达到最大轮次）", tool_log


# ════════════════════════════════════════════
# 第四步：FastAPI 外壳（本课新知识）
# ════════════════════════════════════════════
app = FastAPI(
    title="好物优选智能客服 API",
    description="第二十六课：把客服 Agent 包装成生产级 HTTP API",
    version="1.0.0",
)

# Pydantic 模型：声明请求/响应结构，FastAPI 自动做类型校验
# （多传少传字段、类型不对，直接返回 422 错误，不用手写校验代码）
class ChatRequest(BaseModel):
    session_id: str = Field(..., examples=["user-9527"], description="会话ID，同一个ID共享对话记忆")
    message: str = Field(..., min_length=1, examples=["订单100001到哪了"], description="用户消息")


class ChatReply(BaseModel):
    reply: str = Field(..., description="客服回答")
    tool_log: list = Field(..., description="本轮工具调用链路（Agent 思考白盒化）")


# 会话表：session_id → messages（HTTP 无状态，记忆存在服务端这张表里）
# 演示用内存字典（服务重启就丢）；生产换 Redis + 过期时间，道理相同
SESSIONS: dict[str, list] = {}


@app.get("/health")
def health():
    """健康检查：生产环境负载均衡/容器探活都靠它，返回 ok 说明服务活着"""
    return {"status": "ok", "sessions": len(SESSIONS)}


@app.post("/chat", response_model=ChatReply)
def chat(req: ChatRequest) -> ChatReply:
    # 记忆核心：同 session_id 取同一份历史；首次出现就初始化 system
    messages = SESSIONS.setdefault(
        req.session_id, [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    messages.append({"role": "user", "content": req.message})

    reply, tool_log = run_agent(messages)  # 内核：内部会把 assistant/tool 消息追加进 messages

    return ChatReply(reply=reply, tool_log=tool_log)


if __name__ == "__main__":
    # uvicorn：ASGI 服务器，真正监听端口处理 HTTP 的进程（FastAPI 只是应用）
    uvicorn.run(app, host="127.0.0.1", port=8000)
