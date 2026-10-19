r"""
进阶第二课：错误重试 + 超时控制 + 降级兜底
=============================================
上一课学会了异步并发——100 个请求同时发出去，快了 50 倍。
但现实是：100 个请求里总有几个会失败（网络抖动、API 限流、服务端超时）。
同步代码里一个失败整个程序崩；异步批量里一个失败 gather 整个炸。
这一课教你怎么扛住这些失败，让应用"稳如老狗"。

三道防线：
  1. 重试（tenacity）——网络抖动是暂时的，重试几次就好了
  2. 超时（asyncio.wait_for）——请求不能无限等，超时就掐断
  3. 降级（try/except）——重试用完了还失败，返回兜底回答，别让用户看到报错

本课跑 3 个场景：
  场景1：正常调用（带重试保护，一次成功）——展示"有保护但不影响正常路径"
  场景2：极短超时——故意 timeout=0.001 触发超时降级
  场景3：错误 API key——永久失败，重试耗尽后降级兜底

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\17_retry_timeout_fallback.py
"""
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep,
)

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4-flash"

# 正常客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# 故意配置错误的客户端（场景3 用，保证永久失败）
bad_client = AsyncOpenAI(api_key="sk-invalid-key-12345", base_url=BASE_URL)


# ────────────────────────────────────────────
# 第一道防线：重试（tenacity 装饰器）
# ────────────────────────────────────────────
# @retry 装饰器：把普通函数变成"失败自动重试"的函数
# 三个关键参数：
#   stop=stop_after_attempt(3)  ——最多试 3 次（含首次），超过就抛异常
#   wait=wait_exponential(...)   ——指数退避：第1次等1秒、第2次等2秒...越来越慢
#   retry=retry_if_exception_type(Exception)——什么异常都重试
#   before_sleep=...            ——每次重试前打印日志（方便看到重试过程）
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: print(
        f"    [tenacity] 第{rs.attempt_number}次失败，{rs.next_action.sleep:.1f}秒后重试..."
    ),
)
async def ask_llm(llm_client, question: str) -> str:
    """带重试保护的 LLM 调用——网络抖动自动重试"""
    resp = await llm_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content


# ────────────────────────────────────────────
# 第二道防线：超时控制（asyncio.wait_for）
# ────────────────────────────────────────────
async def ask_with_timeout(llm_client, question: str, timeout_sec: float = 10) -> str:
    """
    给 LLM 调用加超时上限。
    asyncio.wait_for(协程, timeout) ——如果 timeout 秒内没完成，抛 TimeoutError
    防止请求无限挂起。
    """
    return await asyncio.wait_for(
        ask_llm(llm_client, question),
        timeout=timeout_sec,
    )


# ────────────────────────────────────────────
# 第三道防线：降级兜底（try/except）
# ────────────────────────────────────────────
async def ask_with_fallback(llm_client, question: str, timeout_sec: float = 10) -> str:
    """
    三道防线全副武装：
      1. 超时控制——不能无限等
      2. 重试保护——网络抖动自动重试
      3. 降级兜底——全失败了返回兜底回答，不让用户看到报错
    """
    try:
        return await ask_with_timeout(llm_client, question, timeout_sec)
    except asyncio.TimeoutError:
        print("    [降级] 请求超时，返回兜底回答")
        return "【系统繁忙】AI 响应超时，请稍后重试。"
    except Exception as e:
        print(f"    [降级] 重试用完仍失败: {type(e).__name__}")
        return "【系统降级】AI 服务暂时不可用，已为您记录问题，请联系人工客服。"


# ────────────────────────────────────────────
# 主流程：3 个场景对比
# ────────────────────────────────────────────
async def main():
    print("=" * 55)
    print("  进阶第二课：错误重试 + 超时控制 + 降级兜底")
    print("=" * 55)

    question = "用一句话解释什么是递归"

    # ── 场景1：正常调用（带重试保护，一次成功）──
    print("\n【场景1】正常调用（有重试保护但不触发）")
    answer = await ask_with_fallback(client, question)
    print(f"  回答：{answer[:60]}...")

    # ── 场景2：极短超时——故意 timeout=0.001 秒 ──
    print("\n【场景2】极短超时触发降级（timeout=0.001秒）")
    answer = await ask_with_fallback(client, question, timeout_sec=0.001)
    print(f"  结果：{answer}")

    # ── 场景3：错误 API key——永久失败，重试耗尽后降级 ──
    print("\n【场景3】错误 API key（重试3次全失败 → 降级）")
    answer = await ask_with_fallback(bad_client, question)
    print(f"  结果：{answer}")

    print("\n" + "=" * 55)
    print("  三道防线：重试扛住临时故障，超时防止无限等，降级保住用户体验")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
