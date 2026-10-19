"""三个业务工具 + Function Calling 声明

【模块边界】
- 这里只有"工具是什么"：函数本体 + JSON Schema 声明
- agent.py 负责"什么时候调工具"（Agent 循环逻辑）
- config.py 负责"怎么连向量库"（依赖注入）

ORDERS mock 字典后续会换真实数据库查询（第 ② 课 Redis/PostgreSQL）
"""
import json
from .config import collection, get_embedding


# ── Mock 订单数据（第十四课原样，后续换 DB 查询） ──
ORDERS = {
    "100001": {"status": "已发货", "carrier": "顺丰速运", "tracking": "SF1234567890", "eta": "预计明天送达"},
    "100002": {"status": "待发货", "carrier": None, "tracking": None, "eta": "预计今天内发出"},
    "100003": {"status": "已签收", "carrier": "圆通速递", "tracking": "YT9876543210", "eta": "已于昨天签收"},
    "100004": {"status": "已退款", "carrier": None, "tracking": None, "eta": "退款已到账"},
}


def search_faq(question: str) -> dict:
    """检索 FAQ 知识库（Chroma 向量检索 top-3）"""
    q_vec = get_embedding(question)
    results = collection.query(query_embeddings=[q_vec], n_results=3)
    faqs = [
        {"content": d, "distance": round(x, 3)}
        for d, x in zip(results["documents"][0], results["distances"][0])
    ]
    return {"count": len(faqs), "faqs": faqs}


def check_order(order_id: str) -> dict:
    """查询订单状态（mock，后续换 DB）"""
    if order_id in ORDERS:
        return {"order_id": order_id, **ORDERS[order_id]}
    return {"error": f"未找到订单 {order_id}，请确认订单号是否正确"}


def transfer_human(reason: str) -> dict:
    """转人工客服（返回交接信息，本身不"转"）"""
    return {"transfer": True, "reason": reason, "message": "已为您接入人工客服，排队中预计等待1分钟"}


# ── Function Calling 声明（JSON Schema） ──
# 模型"看见"的就是这份声明，description 即 prompt 工程
# 关键教训：description 边界不清 → 模型乱调工具（第十五/二十六课都踩过）
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

tool_map = {
    "search_faq": search_faq,
    "check_order": check_order,
    "transfer_human": transfer_human,
}
