r"""
第三课：Prompt 工程
同一个模型（glm-4-flash），换不同写法的 prompt，输出天差地别。
演示三招：system 角色、要求 JSON 输出、Few-shot 示例。
运行（在 ai-learning 目录下）：
    .venv\Scripts\python.exe 02_prompt_engineering.py
"""
import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


def ask(messages, model="glm-4-flash"):
    """发起一次对话，返回模型生成的文本。messages 决定模型怎么回。"""
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


# ────────────────────────────────────────────
# 招式一：system 角色——给模型定规则，剪掉废话
# 同一个问题：不加 system vs 加 system
# ────────────────────────────────────────────
print("=" * 56)
print("招式一：system 角色——给模型定规则，剪掉废话")
print("=" * 56)

question = "把这句话翻译成英文：今天天气真好"

# 1a. 不加 system：模型爱加“翻译是：”这类前缀和解释
r1 = ask([{"role": "user", "content": question}])
print("【不加 system】")
print(r1)
print("-" * 56)

# 1b. 加 system：明确“只输出译文，不要任何解释/前缀”
r2 = ask([
    {"role": "system", "content": "你是专业翻译。只输出译文本身，不要解释、不要前缀、不要引号。"},
    {"role": "user", "content": question},
])
print("【加 system 约束】")
print(r2)

# ────────────────────────────────────────────
# 招式二：要求 JSON 输出——让结果能被程序解析
# 这是 AI 应用最关键的技能：模型输出能直接进你的代码
# ────────────────────────────────────────────
print()
print("=" * 56)
print("招式二：要求 JSON 输出——让结果能被程序解析")
print("=" * 56)

extract_q = "从这句话里提取姓名和年龄：张三今年28岁，在北京工作。"
r3 = ask([
    {"role": "system", "content": "你是信息提取器。只输出合法 JSON，字段为 name 和 age，不要 markdown 代码块、不要解释。"},
    {"role": "user", "content": extract_q},
])
print("模型输出：", r3)

# 模型有时会在 JSON 外裹一层 ```json 或加文字，用正则抓第一个 {...} 更稳
match = re.search(r"\{.*\}", r3, re.DOTALL)
if match:
    data = json.loads(match.group(0))
    print("程序解析成功 ✅ ->", data, "（类型:", type(data).__name__ + "）")
    print("  现在 data['name'] =", data.get("name"), " data['age'] =", data.get("age"))
else:
    print("没找到 JSON ❌")

# ────────────────────────────────────────────
# 招式三：Few-shot 示例——给几个例子，模型照着格式模仿
# 用 role="assistant" 放“示范回答”，比干巴巴讲规则更管用
# ────────────────────────────────────────────
print()
print("=" * 56)
print("招式三：Few-shot 示例——给例子让模型照格式模仿")
print("=" * 56)

r4 = ask([
    {"role": "system", "content": "对用户评论做情感分类，只输出标签：positive / negative / neutral"},
    # 示例 1
    {"role": "user", "content": "这个产品太好用了，强烈推荐！"},
    {"role": "assistant", "content": "positive"},
    # 示例 2
    {"role": "user", "content": "包装破烂，物流慢，差评。"},
    {"role": "assistant", "content": "negative"},
    # 真正要分类的
    {"role": "user", "content": "东西收到了，还没拆，看着一般。"},
])
print("分类结果：", r4)
