r"""
第十课：第一个真 Agent
第四课我们学了 Function Calling：模型说要调工具→我们执行→喂回去→模型再答。
但第四课只调了"一次"就结束——那还不算 Agent。

真正的 Agent 长什么样？答：它是一个【循环】。
模型自己决定：
  - 这一轮要不要调工具？调几个？
  - 调完拿到结果后，要不要再调下一个工具？
  - 什么时候信息够了，可以停下来给最终答案？
你的代码只负责：循环→执行工具→喂回结果，直到模型不再要工具为止。

本课给模型 3 个工具：
  1. get_time()         —— 查当前时间
  2. get_weather(city)  —— 查城市天气
  3. calculate(expr)    —— 算术计算

然后问一个需要多步的问题，看模型自己编排工具调用顺序。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\09_first_agent.py
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
# 第 1 步：定义 3 个本地工具函数
# 真实项目里这些函数会调外部 API/数据库，这里用 mock 演示
# ────────────────────────────────────────────
def get_time() -> dict:
    """返回当前时间"""
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": now.hour,
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
    }


def get_weather(city: str) -> dict:
    """查城市天气（mock 数据）"""
    fake = {
        "北京": {"temp": 22, "weather": "晴", "wind": "北风3级"},
        "上海": {"temp": 26, "weather": "多云", "wind": "东南风2级"},
        "广州": {"temp": 30, "weather": "雷阵雨", "wind": "南风2级"},
    }
    return fake.get(city, {"temp": 25, "weather": "未知", "wind": "微风"})


def calculate(expression: str) -> dict:
    """安全算术计算（只允许数字和+-*/）"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": "表达式含非法字符"}
    try:
        result = eval(expression)  # 已做字符白名单校验，真实项目可用 ast 安全解析
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


# ────────────────────────────────────────────
# 第 2 步：把 3 个工具用 JSON Schema 声明给模型
# ────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前日期和时间。当用户问现在几点、今天星期几、日期时调用。",
            "parameters": {"type": "object", "properties": {}},  # 无参数
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

# 工具名 → Python 函数 的映射，方便循环里统一调用
tool_map = {
    "get_time": get_time,
    "get_weather": get_weather,
    "calculate": calculate,
}


# ────────────────────────────────────────────
# 第 3 步：Agent 主循环 —— 本课核心
# 和第四课区别：第四课调一次工具就停；这里用 while 循环，
# 模型可以连续调多轮工具，直到它自己觉得够了
# ────────────────────────────────────────────
def run_agent(user_msg: str, max_turns: int = 10) -> tuple:
    """
    运行 Agent，返回 (最终回答, 实际用了几轮)。
    max_turns 防止模型无限循环（安全阀）。
    """
    # system 必须强调"只能调工具、禁止编造"——glm-4-flash 不加会自己编数据
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个能调用工具的助手。回答任何事实性问题（时间、天气、计算）"
                "时必须调用对应工具获取真实数据，禁止凭空编造。"
                "拿到所有需要的信息后，用自然语言简短回答用户。"
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    for turn in range(1, max_turns + 1):
        # 每一轮都把完整对话历史 + 工具声明发给模型
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # 模型自己决定要不要调、调哪个
        )
        msg = resp.choices[0].message

        # 关键判断：模型这一轮是想调工具，还是直接给答案？
        if not msg.tool_calls:
            # 模型不再要工具 → 它给出了最终答案 → 退出循环
            print(f"\n[第{turn}轮] 模型给出最终答案，循环结束。")
            return msg.content, turn

        # 模型要调工具。先把模型这一轮的回复追加进历史
        messages.append(msg)
        print(f"\n[第{turn}轮] 模型决定调用 {len(msg.tool_calls)} 个工具：")

        # 执行它要调的所有工具（可能一轮调多个！）
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"   → {name}({args})")

            # 执行真正的 Python 函数
            result = tool_map[name](**args)
            print(f"     返回：{result}")

            # 用 role="tool" 把结果喂回去，tool_call_id 用来配对
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        # 工具结果喂回去后，进入下一轮循环，让模型继续判断

    return "（达到最大轮次仍未完成）", max_turns


# ────────────────────────────────────────────
# 演示：问一个需要多步工具的问题
# ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("第十课：第一个真 Agent")
    print("=" * 60)

    question = "现在几点？今天星期几？再帮我算一下 15 乘 23 加 100 等于多少，还有查一下北京天气，出门要带伞吗？"

    print(f"\n用户问题：{question}")
    print("-" * 60)
    print("Agent 开始思考（每轮打印它调了什么工具）：")

    answer, turns = run_agent(question)

    print("-" * 60)
    print(f"Agent 共执行了 {turns} 轮")
    print(f"\n最终回答：\n{answer}")
