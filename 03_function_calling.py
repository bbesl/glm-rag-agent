r"""
第四课：Function Calling（函数调用）
第三课我们在 prompt 里手动要 JSON，然后自己解析。
这一课更进一步：把"工具"正式声明给模型，模型自己决定该不该调、调哪个、传什么参数，
你的代码执行完工具后，把结果喂回去，模型再给出最终回答。
这是通往 Agent 的关键一跃——模型不再只会说话，它开始"用工具"。

场景：查天气。模型本身不知道今天天气，但我们会给它一个 get_weather 工具。
运行（在 ai-learning 目录下）：
    .venv\Scripts\python.exe 03_function_calling.py
"""
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


# ────────────────────────────────────────────
# 第 1 步：定义一个本地函数
# 注意：模型不会真的"执行"代码，执行权永远在你手里
# 这里用 mock 数据模拟天气查询，真实项目里这一步会调外部天气 API
# ────────────────────────────────────────────
def get_weather(city: str) -> dict:
    """假装查到了天气（真实项目里这里会请求天气 API）"""
    fake_data = {
        "北京": {"temp": 22, "weather": "晴", "wind": "北风3级"},
        "上海": {"temp": 26, "weather": "多云", "wind": "东南风2级"},
        "广州": {"temp": 30, "weather": "雷阵雨", "wind": "南风2级"},
    }
    return fake_data.get(city, {"temp": 25, "weather": "未知", "wind": "微风"})


# ────────────────────────────────────────────
# 第 2 步：用 JSON Schema 把这个工具"描述"给模型
# 模型只看这段描述来决定要不要调、传什么参数——它看不到上面的函数体
# description 写清楚，模型才知道什么时候该用它
# ────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气。当用户问天气、温度、是否下雨等问题时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "要查询天气的城市名，如'北京'、'上海'",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

# 一个字典把工具名映射到真正的 Python 函数，方便后面调用
tool_map = {"get_weather": get_weather}


def ask_with_tools(user_msg):
    """完整的 Function Calling 流程，返回最终的自然语言回答"""
    # system 明确要求"必须调工具、禁止编造"，让免费模型稳定走 Function Calling 流程
    messages = [
        {"role": "system", "content": "你是天气助手。用户问天气时必须调用 get_weather 工具拿真实数据，禁止自己编造温度或天气状况。拿到结果后用自然语言简短回答。"},
        {"role": "user", "content": user_msg},
    ]

    # 第 3 步：第一轮调用——把 tools 一起发给模型
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        tools=tools,          # 关键：把工具声明传进去
        tool_choice="auto",   # auto = 模型自己决定要不要调、调哪个
    )
    msg = resp.choices[0].message

    # 第 4 步：检查模型是想直接回答，还是想调工具
    if not msg.tool_calls:
        # 模型觉得不需要工具，直接给了答案
        return msg.content

    # 模型决定调工具！把模型这一轮的回答追加进对话历史
    messages.append(msg)

    # 第 5 步：执行模型要调的工具，把结果喂回去
    for call in msg.tool_calls:
        name = call.function.name                      # 工具名，如 "get_weather"
        args = json.loads(call.function.arguments)     # 参数 JSON，如 {"city": "北京"}
        print(f"  [模型要调工具] {name}({args})")

        result = tool_map[name](**args)                # 真正执行你的 Python 函数
        print(f"  [函数返回] {result}")

        # 用 role="tool" 把执行结果发回去，tool_call_id 用来配对是哪个调用
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    # 第 6 步：第二轮调用——模型拿到工具结果，生成最终的自然语言回答
    final = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        tools=tools,
    )
    return final.choices[0].message.content


# ────────────────────────────────────────────
# 演示：带工具 vs 不带工具，看模型行为差异
# ────────────────────────────────────────────
print("=" * 56)
print("对比一：不带工具——模型不知道今天天气，只能瞎编或承认不知道")
print("=" * 56)
resp = client.chat.completions.create(
    model="glm-4-flash",
    messages=[{"role": "user", "content": "北京今天多少度？要带伞吗？"}],
)
print("模型回答：", resp.choices[0].message.content)
print("（注意：模型根本没查天气，温度和是否下雨都是它编的或拒答的）")

print()
print("=" * 56)
print("对比二：带工具——模型自己决定调 get_weather，拿真实数据再回答")
print("=" * 56)
answer = ask_with_tools("北京今天多少度？要带伞吗？")
print()
print("最终回答：", answer)
