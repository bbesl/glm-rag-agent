r"""
进阶第一课：异步并发调用 LLM（asyncio + AsyncOpenAI）
=====================================================
基础 15 课里，每次调 LLM 都是"发请求 → 等响应 → 拿结果"——同步阻塞。
当你只需调 1 次，等 2 秒无所谓。
但真实业务常要批量：一次性给 100 条评论分类、批量翻译、批量摘要……
串行的问题就暴露了：5 个请求 × 2 秒/个 = 10 秒，全部排队等。

异步并发的核心思路：
  同步：发A→等A→发B→等B→发C→等C    总耗时 = A+B+C（累加）
  异步：发A→发B→发C→等全部回来      总耗时 ≈ max(A,B,C)（最慢那个）

LLM 调用是典型的 I/O 密集型任务（CPU 几乎不干活，全在等网络），
正是异步并发最擅长的场景。

本课对比：同样 5 个问题，同步串行 vs 异步并发，看耗时差距。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\16_async_llm.py
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4-flash"

# 同步客户端（前面 15 课一直在用的）
sync_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 异步客户端（本课新主角）——同一个 openai 库，换个类名就是异步版
async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)


# 5 个测试问题（模拟批量处理场景：给一批编程概念各生成一句话解释）
QUESTIONS = [
    "用一句话解释什么是递归",
    "用一句话解释什么是闭包",
    "用一句话解释什么是生成器",
    "用一句话解释什么是装饰器",
    "用一句话解释什么是上下文管理器",
]


# ────────────────────────────────────────────
# 同步版：一个个串行调用
# ────────────────────────────────────────────
def ask_sync(question: str) -> str:
    """同步版：发一个请求，阻塞等结果，期间程序什么也干不了"""
    resp = sync_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content


def run_sync_batch() -> float:
    """同步批量：串行调用 5 次，总耗时 = 5 次累加"""
    print("\n【同步版】串行调用 5 个问题...")
    start = time.time()
    for i, q in enumerate(QUESTIONS, 1):
        t0 = time.time()
        answer = ask_sync(q)
        dt = time.time() - t0
        print(f"  第{i}个完成（{dt:.2f}秒）：{answer[:40]}...")
    total = time.time() - start
    print(f"  >>> 同步总耗时：{total:.2f} 秒")
    return total


# ────────────────────────────────────────────
# 异步版：用 asyncio.gather 并发调用
# ────────────────────────────────────────────
async def ask_async(question: str) -> str:
    """
    异步版：await 表示"等这个请求的响应"，
    但在等待期间，事件循环会切去执行别的协程——这就是并发。
    """
    resp = await async_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content


async def run_async_batch() -> float:
    """异步批量：用 asyncio.gather 同时发起 5 个请求，等全部回来"""
    print("\n【异步版】并发调用 5 个问题...")
    start = time.time()
    # 关键行：asyncio.gather 把多个协程"同时"启动
    # 5 个请求几乎同时发出，不用排队
    # 返回值顺序与传入顺序一一对应
    results = await asyncio.gather(*[ask_async(q) for q in QUESTIONS])
    total = time.time() - start
    for i, ans in enumerate(results, 1):
        print(f"  第{i}个完成：{ans[:40]}...")
    print(f"  >>> 异步总耗时：{total:.2f} 秒")
    return total


# ────────────────────────────────────────────
# 主流程：先同步后异步，对比耗时
# ────────────────────────────────────────────
async def main():
    print("=" * 55)
    print("  进阶第一课：异步并发调用 LLM")
    print("=" * 55)

    # 先跑同步版
    sync_time = run_sync_batch()

    # 再跑异步版（run_async_batch 本身是协程，要 await）
    async_time = await run_async_batch()

    # 对比
    print("\n" + "=" * 55)
    print(f"  同步串行：{sync_time:.2f} 秒")
    print(f"  异步并发：{async_time:.2f} 秒")
    speedup = sync_time / async_time if async_time > 0 else 0
    print(f"  加速比  ：{speedup:.1f}x")
    print("=" * 55)


if __name__ == "__main__":
    # asyncio.run 是异步程序的入口：创建事件循环，跑完 main 协程，关闭循环
    asyncio.run(main())
