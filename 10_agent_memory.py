r"""
第十一课：给 Agent 加对话记忆
第十课的 Agent 每次运行都是"一次性"的——问完就忘，下次问不记得上句。
真正的助手得有记忆：你上一句说"北京"，下一句说"那上海呢"，它要懂"呢"指天气。

记忆的本质其实特别简单：messages 列表跨轮不重置。
第十课每次 run_agent 都新建 messages；这一课把 messages 提到外层 while 循环之外，
让它一直累积——用户的话、模型的话、工具结果，全都留在列表里。
模型每次调用都能看到完整历史，于是它就"记得"了。

本课双层循环结构：
  外层 while True + input()  —— 多轮对话（第九课的思路）
  内层 run_agent 的 for 循环 —— 单轮内多步工具调用（第十课的思路）

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\10_agent_memory.py
"""
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


# ────────────────────────────────────────────
# 工具函数（和第十课相同，复用）
# ────────────────────────────────────────────
def get_time() -> dict:
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": now.hour,
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
    }


def get_weather(city: str) -> dict:
    fake = {
        "北京": {"temp": 22, "weather": "晴", "wind": "北风3级"},
        "上海": {"temp": 26, "weather": "多云", "wind": "东南风2级"},
        "广州": {"temp": 30, "weather": "雷阵雨", "wind": "南风2级"},
    }
    return fake.get(city, {"temp": 25, "weather": "未知", "wind": "微风"})


def calculate(expression: str) -> dict:
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": "表达式含非法字符"}
    try:
        return {"expression": expression, "result": eval(expression)}
    except Exception as e:
        return {"error": str(e)}


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前日期和时间。当用户问现在几点、今天星期几、日期时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气。当用户问天气、温度、是否下雨时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如'北京'、'上海'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。当用户要做算术运算（加减乘除）时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '15*23+100'"},
                },
                "required": ["expression"],
            },
        },
    },
]

tool_map = {"get_time": get_time, "get_weather": get_weather, "calculate": calculate}


# ────────────────────────────────────────────
# run_agent：在【已有对话历史】上跑一轮 Agent
# 和第十课区别：messages 由调用方传入（含历史），不再函数内新建
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
            # 关键：把模型的回答也追加进历史，这样下一轮对话它才"记得自己说过什么"
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
        # 喂回后回到循环开头，模型继续判断

    return "（达到最大轮次）", max_turns


# ────────────────────────────────────────────
# 主循环：交互式对话 Agent（带记忆）
# ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("第十一课：带记忆的交互式 Agent")
    print("输入 quit / 退出 / q 结束对话")
    print("=" * 60)

    # 记忆的核心：messages 在 while 之外初始化，跨轮累积不重置
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个能调工具的助手。回答事实性问题（时间、天气、计算）"
                "时必须调用对应工具，禁止编造。注意对话上下文，用户说"
                "'那XX呢'指的是上一句问过的同类型信息。回答要简短自然。"
            ),
        }
    ]

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            # 管道喂入测试时，输入读完会触发 EOFError，正常退出
            print("\n（输入结束，退出）")
            break

        if user_input.lower() in ("quit", "退出", "q", "exit"):
            print("再见！")
            break
        if not user_input:
            continue

        # 把用户这一句追加进历史
        messages.append({"role": "user", "content": user_input})

        print("思考中...", end="\r")
        answer, turns = run_agent(messages)

        print(f"助手（本轮内循环 {turns} 轮）：{answer}")
        # answer 已经在 run_agent 里作为 assistant 消息 append 进 messages 了


if __name__ == "__main__":
    main()
