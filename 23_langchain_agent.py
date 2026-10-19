r"""
第二十三课：用 LangChain 重写客服 Agent
========================================
第十四课我们手写了 40 行 Agent 循环（create → 检查 tool_calls → 执行工具
→ role=tool 喂回 → 继续循环）。本课用 LangChain 重写同样的客服 Agent，
对比看框架帮你省了什么、藏了什么。

【新旧对照表】
  手写版（13_customer_service_agent.py）    LangChain 版（本课）
  ─────────────────────────────────────    ─────────────────────
  client = OpenAI(base_url=...)        →   ChatOpenAI(base_url=...)
  tools = [{JSON Schema 声明}]         →   @tool 装饰器（docstring 即 description）
  tool_map = {名字: 函数}              →   不需要（框架自动映射）
  run_agent 40 行 while 循环           →   create_agent(...) 一行
  messages 手动 append/接力            →   返回的 messages 直接接力
  Chroma 原生 query + 手算 embedding   →   Chroma.as_retriever() 自动向量化

【不变的铁律】
  system prompt"必须调工具禁止编造"原样保留——框架再好，
  glm-4-flash 不加这句照样编数据（第四课踩的坑跟到这课）。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\23_langchain_agent.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# LangChain 三大件：Agent 工厂 / Chroma 封装 / 消息与工具
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ────────────────────────────────────────────
# 第一步：模型——ChatOpenAI 等价于手写版的 OpenAI 客户端
# 参数和手写版一样：model / api_key / base_url，兼容协议的好处
# ────────────────────────────────────────────
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0,
)


# ────────────────────────────────────────────
# 第二步：FAQ 向量库——复用第十三课持久化的 .chroma_faq
# OpenAIEmbeddings 自动调智谱 embedding-3 把查询转向量
# as_retriever() 返回检索器：retriever.invoke(问题) → 最相关的 k 条 Document
# ────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    # 关键：默认 True 时 tiktoken 把文本切成 token id 再发请求——
    # 那是 OpenAI 自家的预处理，智谱 embedding-3 不认，检索结果会乱。
    # False = 文本原样发送（第八课手写版就是这么调的）
    check_embedding_ctx_length=False,
)
vectorstore = Chroma(
    collection_name="faq_customer_service",
    persist_directory=str(BASE_DIR / ".chroma_faq"),
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ────────────────────────────────────────────
# 第三步：定义工具——@tool 装饰器替代手写版两样东西：
#   ① JSON Schema 声明（函数签名+类型注解自动生成 schema）
#   ② description（docstring 就是模型看到的工具描述，第十五课的教训：
#      边界描述不清模型会乱调，所以 docstring 照抄调优后的版本）
# 返回值统一 str——模型直接读文本，比手写版返回 dict 更省事
# ────────────────────────────────────────────
@tool
def search_faq(question: str) -> str:
    """检索客服FAQ知识库。当用户问发货时间、退货流程、支付方式、保修期、发票等不需要订单号的通用问题时调用。"""
    docs = retriever.invoke(question)
    return "\n---\n".join(d.page_content for d in docs)


# mock 订单数据（照抄第十四课）
ORDERS = {
    "100001": {"status": "已发货", "carrier": "顺丰速运", "tracking": "SF1234567890", "eta": "预计明天送达"},
    "100002": {"status": "待发货", "carrier": None, "tracking": None, "eta": "预计今天内发出"},
    "100003": {"status": "已签收", "carrier": "圆通速递", "tracking": "YT9876543210", "eta": "已于昨天签收"},
    "100004": {"status": "已退款", "carrier": None, "tracking": None, "eta": "退款已到账"},
}


@tool
def check_order(order_id: str) -> str:
    """查询订单状态和物流信息。必须用户提供具体订单号才能调用，如 '100001'。"""
    if order_id in ORDERS:
        info = ORDERS[order_id]
        carrier = info["carrier"] or "暂无"
        tracking = info["tracking"] or "暂无"
        return f"订单{order_id}：状态={info['status']}，快递={carrier}，单号={tracking}，{info['eta']}"
    return f"未找到订单 {order_id}，请确认订单号是否正确"


@tool
def transfer_human(reason: str) -> str:
    """转接人工客服。当用户明确要求人工、要投诉、或问题超出FAQ与订单范围时调用。"""
    return f"已为您接入人工客服，排队中预计等待1分钟（转人工原因：{reason}）"


# ────────────────────────────────────────────
# 第四步：create_agent 一行替代手写版 40 行循环
# system_prompt 照抄第十四课——"禁止编造"铁律不变
# 内部其实就是 LangGraph 帮你跑"create→调工具→喂回→继续"那个循环
# ────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是「好物优选」电商平台的智能客服。工具：search_faq（查FAQ政策）、"
    "check_order（查订单状态，需订单号）、transfer_human（转人工）。\n"
    "规则：\n"
    "1) 涉及平台业务（发货/物流/退换货/退款/支付/发票/保修/订单进度）必须先调工具"
    "拿真实数据再回答，禁止编造订单号、物流单号、政策，禁止凭常识估计；\n"
    "2) 用户说'那个订单''这个订单的快递单号'时，订单号在对话历史里，"
    "找出后直接调 check_order，不要反问；\n"
    "3) 用户要人工/投诉/超出范围用 transfer_human；\n"
    "4) 语气专业、简洁、有温度；简单寒暄不必调工具。"
)

agent = create_agent(
    model=llm,
    tools=[search_faq, check_order, transfer_human],
    system_prompt=SYSTEM_PROMPT,
)


# ────────────────────────────────────────────
# 第五步：多轮对话——外层 input() 循环（第九课）不变
# 记忆 = history 接力：每次把 Agent 返回的完整 messages
# （用户问题+模型思考+工具结果+最终回答）赋回 history
# stream 逐条打印中间步骤 = Agent 思考白盒化（第十五课 st.status 的思路）
# ────────────────────────────────────────────
def chat_loop():
    history = []
    print("客服小智上线（输入 quit/退出/q 结束）")
    while True:
        try:
            q = input("\n你：").strip()
        except EOFError:
            break
        if not q or q.lower() in ("quit", "退出", "q"):
            break

        history.append(HumanMessage(content=q))
        answer = ""
        for step in agent.stream({"messages": history}, stream_mode="values"):
            last = step["messages"][-1]
            if last.type == "ai" and getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    print(f"   [调工具] {tc['name']}({tc['args']})")
            elif last.type == "tool":
                preview = str(last.content)[:40].replace("\n", " ")
                print(f"   [工具返回] {preview}...")
            elif last.type == "ai":
                answer = last.content

        if not answer:  # 兜底：异常截断时取最后一条
            answer = str(step["messages"][-1].content)
        print(f"客服：{answer}")

        # 关键一行：完整历史接力（含中间工具消息），下一轮模型记得一切
        history = step["messages"]


if __name__ == "__main__":
    chat_loop()
