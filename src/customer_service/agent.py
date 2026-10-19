"""Agent 核心：run_agent 循环 + SYSTEM_PROMPT

【模块边界】
- 这里只有"Agent 是什么"：循环逻辑 + 系统提示词
- 工具本体在 tools.py，API 客户端在 config.py
- FastAPI 路由在 main.py（谁来调 run_agent 不在这关心）

设计决策：Agent 循环对配置和工具的依赖都通过 import 拿，
不做 DI 容器（单服务场景没必要过度工程化）。
"""
import json
from .config import client
from .tools import tools, tool_map


# ── 系统提示词（第十四课铁律原样保留） ──
# 关键教训：glm-4-flash 不加"禁止编造"会假装调工具自己编答案
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


def run_agent(messages: list, max_turns: int = 10) -> tuple[str, list]:
    """Agent 循环：模型自主决策调几次工具，直到给出最终答案或达到安全阀

    Args:
        messages: 消息历史列表（含 system、user、assistant、tool 消息）
        max_turns: 最大工具调用轮次（安全阀，防无限循环）

    Returns:
        (最终回答文本, 本轮工具调用链路日志列表)
    """
    tool_log = []
    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model="glm-4-flash", messages=messages, tools=tools, tool_choice="auto"
        )
        msg = resp.choices[0].message

        # 无 tool_calls → 模型给出最终答案，循环结束
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content or "（模型未返回内容）", tool_log

        # 有 tool_calls → 执行工具，结果喂回继续循环
        # 以纯 dict 追加（不能直接 append SDK 的 ChatCompletionMessage 对象：
        # Redis/PostgreSQL 存会话需要 JSON 可序列化，pydantic 对象序列化会 TypeError）
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in msg.tool_calls
            ],
        })
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
