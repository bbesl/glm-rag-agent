r"""
第十二课：多 Agent 协作
前两课的 Agent 都是"一个人干所有事"——一个 Agent 拿着全部工具、一个提示词、一个循环。
工具一多，提示词会臃肿，模型也容易乱调。

多 Agent 的思路：分工。
  - 主 Agent（协调者）：不直接碰底层工具，只负责"拆任务、分派给谁、汇总结果"
  - 子 Agent（专员）：每个有自己的专长工具和提示词，只管自己那摊事

关键技巧：把每个子 Agent 包装成主 Agent 的一个"工具"。
主 Agent 调用 sub_agent 工具 → 触发子 Agent 内部跑一整套循环 → 返回答案文本 → 主 Agent 喂回汇总。

于是形成两层结构：
  主 Agent 循环（调子Agent） → 子 Agent 循环（调底层工具）

本课两个子 Agent：
  - info_agent：信息专员，专管时间/天气（拿 get_time、get_weather）
  - math_agent：计算专员，专管算术（拿 calculate）

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\11_multi_agent.py
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
# 底层工具函数（和前两课相同）
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


# 全局工具名→函数映射，子 Agent 按需取用
all_tool_map = {"get_time": get_time, "get_weather": get_weather, "calculate": calculate}


# ────────────────────────────────────────────
# 三个工具的 JSON Schema 声明（拆开，方便子 Agent 各取所需）
# ────────────────────────────────────────────
TOOL_TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "获取当前日期和时间。当用户问现在几点、今天星期几、日期时调用。",
        "parameters": {"type": "object", "properties": {}},
    },
}
TOOL_WEATHER = {
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
}
TOOL_CALC = {
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
}


# ────────────────────────────────────────────
# 通用子 Agent 循环：复用第十课的 while 逻辑
# 区别：tools 由调用方指定（每个子 Agent 用自己的工具子集）
# ────────────────────────────────────────────
def run_sub_agent(messages: list, sub_tools: list, max_turns: int = 8) -> tuple:
    """子 Agent 内部循环，返回 (最终答案文本, 内循环轮数)。"""
    for turn in range(1, max_turns + 1):
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            tools=sub_tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content, turn

        messages.append(msg)
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            result = all_tool_map[name](**args)
            print(f"        [子Agent调工具] {name}({args}) → {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    return "（子Agent超轮次）", max_turns


# ────────────────────────────────────────────
# 子 Agent 1：信息专员（时间 + 天气）
# ────────────────────────────────────────────
INFO_TOOLS = [TOOL_TIME, TOOL_WEATHER]


def info_agent(question: str) -> str:
    """信息专员：专管时间和天气查询。被主 Agent 当工具调用。"""
    print(f"    ▶ 信息专员 info_agent 接到任务：{question}")
    messages = [
        {
            "role": "system",
            "content": "你是信息查询专员，只负责时间和天气。必须调用工具获取真实数据，禁止编造。简短回答。",
        },
        {"role": "user", "content": question},
    ]
    answer, turns = run_sub_agent(messages, INFO_TOOLS)
    print(f"    ◀ 信息专员返回（{turns}轮）：{answer}")
    return answer


# ────────────────────────────────────────────
# 子 Agent 2：计算专员（算术）
# ────────────────────────────────────────────
MATH_TOOLS = [TOOL_CALC]


def math_agent(question: str) -> str:
    """计算专员：专管算术。被主 Agent 当工具调用。"""
    print(f"    ▶ 计算专员 math_agent 接到任务：{question}")
    messages = [
        {
            "role": "system",
            "content": "你是计算专员，只负责算术。必须调用 calculate 工具，禁止自己心算编造。简短回答。",
        },
        {"role": "user", "content": question},
    ]
    answer, turns = run_sub_agent(messages, MATH_TOOLS)
    print(f"    ◀ 计算专员返回（{turns}轮）：{answer}")
    return answer


# ────────────────────────────────────────────
# 主 Agent：把两个子 Agent 当工具
# 注意：主 Agent 看不到 get_time/get_weather/calculate
# 它只能"调子 Agent"，由子 Agent 再去调底层工具——这就是分层
# ────────────────────────────────────────────
MASTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "info_agent",
            "description": "把时间和天气相关的子任务交给信息专员处理。涉及几点、星期几、日期、天气、温度时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要交给信息专员的子问题"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "math_agent",
            "description": "把算术计算相关的子任务交给计算专员处理。涉及加减乘除、算式求值时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要交给计算专员的子问题"},
                },
                "required": ["question"],
            },
        },
    },
]

master_tool_map = {"info_agent": info_agent, "math_agent": math_agent}


def run_master_agent(user_msg: str, max_turns: int = 8) -> tuple:
    """主 Agent 循环：决定调哪个子 Agent，拿到结果后汇总。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是任务协调者。用户的问题可能包含多个子任务，"
                "请把每类子任务分派给对应专员（info_agent 管时间天气，math_agent 管算术）。"
                "拿到所有专员结果后，用自然语言汇总回答用户。"
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    for turn in range(1, max_turns + 1):
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            tools=MASTER_TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # 主 Agent 拿到各专员结果，给出最终汇总
            return msg.content, turn

        messages.append(msg)
        print(f"[主Agent 第{turn}轮] 分派 {len(msg.tool_calls)} 个子任务：")
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"  → 调子Agent {name}({args})")
            # 调子 Agent —— 内部会跑一整套子循环
            result = master_tool_map[name](**args)
            # 子 Agent 返回的是自然语言文本，直接作为 tool 结果喂回主 Agent
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })
        # 子 Agent 结果喂回后，主 Agent 继续判断还要不要调别的子 Agent

    return "（主Agent超轮次）", max_turns


# ────────────────────────────────────────────
# 演示
# ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("第十二课：多 Agent 协作（主 Agent + 子 Agent）")
    print("=" * 64)

    question = "现在几点了？今天星期几？还有帮我算一下 (12+8) 乘 5 等于多少？"
    print(f"\n用户问题：{question}")
    print("-" * 64)

    answer, turns = run_master_agent(question)

    print("-" * 64)
    print(f"主 Agent 共 {turns} 轮")
    print(f"\n最终汇总回答：\n{answer}")
