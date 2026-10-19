r"""
第二课：第一次调用大模型 API
目标：用 10 行代码让智谱 GLM-4-Flash 回你一句话，亲手感受 LLM 到底是什么。
运行方式（在 ai-learning 目录下）：
    .venv\Scripts\python.exe 01_first_call.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 文件读 API Key（密钥永不写进源码，这是标准做法）
load_dotenv()

# 1. 创建客户端：告诉 SDK 用哪个 key、连哪个服务
#    智谱的接口完全兼容 OpenAI 协议，所以直接用 openai 这个库，只改 base_url
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# 2. 发起一次对话：messages 就是你要喂给模型的“输入文本”
#    role="user" 表示这一条是“用户说的话”
response = client.chat.completions.create(
    model="glm-4-flash",  # 智谱的免费模型
    messages=[
        {"role": "user", "content": "用一句话介绍你自己"}
    ],
)

# 3. 从返回结果里取出模型生成的那句话
answer = response.choices[0].message.content
print("模型说：", answer)
