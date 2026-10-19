r"""
进阶第四课（工程化方向最后一课）：速率限制 + 并发控制
=========================================================
第十六课学会了异步并发：asyncio.gather 把 100 个请求同时发出去，快是真快。
但问题来了——API 有 rate limit（速率上限），100 个同时发很可能被服务器拒掉
返回 429 Too Many Requests，直接挂掉。

这一课教两个"限流"武器，让批量请求"快但不超"：

  1. Semaphore（信号量）—— 控制并发上限
     "最多 N 个请求同时在飞行"——就像高速公路收费站只有 5 个窗口，
     最多 5 辆车同时过，其余在后面排队。快的先过，窗口腾出来下一个补上。

  2. Token Bucket（令牌桶）—— 控制发送速率
     "每秒最多发 N 个请求"——像水龙头匀速滴水，不管你桶里攒了多少，
     滴出来的速度是恒定的。适合精确控制"单位时间内的请求数"。

两者的区别：
  Semaphore 控制的是"同时在跑的数"——10个请求限5并发 = 分2批，跑完一批再下一批
  Token Bucket 控制的是"每秒发几个"——10个请求 3/秒 = 约3秒匀速发完

本课对比三种模式跑同样 10 个请求：
  模式1：无限并发（第16课原样，可能被限流）
  模式2：Semaphore 限并发=5（最多5个同时跑）
  模式3：Token Bucket 限速=3/秒（匀速发出）

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\19_rate_limit.py
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4-flash"

# 异步客户端（第十六课的主角，本课继续用）
async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# 10 个测试问题（模拟批量场景：给一批编程概念各生成一句话解释）
QUESTIONS = [
    "用一句话解释什么是递归",
    "用一句话解释什么是闭包",
    "用一句话解释什么是生成器",
    "用一句话解释什么是装饰器",
    "用一句话解释什么是上下文管理器",
    "用一句话解释什么是迭代器",
    "用一句话解释什么是元类",
    "用一句话解释什么是 GIL",
    "用一句话解释什么是协程",
    "用一句话解释什么是异步IO",
]


# ────────────────────────────────────────────
# 基础函数：单个异步请求
# ────────────────────────────────────────────
async def ask_async(question: str) -> str:
    """发一个异步 LLM 请求，返回回答文本"""
    resp = await async_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content


# ════════════════════════════════════════════
# 模式1：无限并发（第十六课原样）
# ════════════════════════════════════════════
async def run_unlimited() -> float:
    """10 个请求全量同时发——最快，但真实场景可能触发 API 限流"""
    print("\n【模式1】无限并发（10个全部同时发）...")
    start = time.time()
    results = await asyncio.gather(*[ask_async(q) for q in QUESTIONS])
    total = time.time() - start
    for i, ans in enumerate(results, 1):
        print(f"  第{i:2d}个：{ans[:35]}...")
    print(f"  >>> 耗时 {total:.2f} 秒")
    print(f"  ⚠ 真实场景：API 限流 10个/秒 时，这种发法可能被拒（429）")
    return total


# ════════════════════════════════════════════
# 模式2：Semaphore 并发上限
# ════════════════════════════════════════════
async def ask_with_semaphore(
    sem: asyncio.Semaphore, question: str, idx: int
) -> str:
    """
    用 Semaphore 包一层：
      async with sem → 获取许可（有名额才能进，没名额就等）
      离开 with 块 → 自动释放许可（腾出名额给下一个）
    """
    async with sem:  # acquire → 进（名额-1）；离开块自动 release（名额+1）
        t0 = time.time()
        ans = await ask_async(question)
        print(f"  第{idx:2d}个完成（{time.time()-t0:.2f}s）")
        return ans


async def run_with_semaphore(limit: int = 5) -> float:
    """限制最多 5 个请求同时在飞行——安全版并发"""
    print(f"\n【模式2】Semaphore 并发上限={limit}（最多{limit}个同时跑）...")
    sem = asyncio.Semaphore(limit)  # 创建信号量：内部计数器初始=limit
    start = time.time()
    # 10 个协程抢 5 个名额：先进去的 5 个同时跑，有人跑完腾名额，下一个补上
    results = await asyncio.gather(
        *[ask_with_semaphore(sem, q, i) for i, q in enumerate(QUESTIONS, 1)]
    )
    total = time.time() - start
    print(f"  >>> 耗时 {total:.2f} 秒（比模式1慢，但不会超限）")
    return total


# ════════════════════════════════════════════
# 模式3：Token Bucket 令牌桶速率限制
# ════════════════════════════════════════════
class TokenBucket:
    """
    令牌桶速率限制器：
      - 按固定 rate（个/秒）往桶里补令牌
      - 桶最多存 capacity 个令牌（满了就不补了，攒着）
      - 每个请求要消耗 1 个令牌才能发出
      - 没令牌就等，等桶补够再发

    效果：不管你攒了多少请求，发出速率恒定在 rate 个/秒。
    """
    def __init__(self, rate: float, capacity: int):
        self.rate = rate              # 每秒补充多少令牌
        self.capacity = capacity      # 桶最大容量
        self.tokens = float(capacity) # 初始满桶
        self.last_refill = time.monotonic()  # 上次补充时间
        self.lock = asyncio.Lock()    # 多协程并发补充要加锁

    async def acquire(self):
        """获取一个令牌：有就直接拿，没有就等到够"""
        async with self.lock:
            # 先按时间差补充令牌（懒补充：只在被请求时算）
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate,
            )
            self.last_refill = now

            if self.tokens >= 1:
                # 有令牌，消耗一个，放行
                self.tokens -= 1
                return

            # 不够 1 个令牌：算要等多久才补充出 1 个
            deficit = 1 - self.tokens       # 缺多少
            wait_time = deficit / self.rate  # 要等多久
            await asyncio.sleep(wait_time)   # 等（持有锁，让请求排队匀速发出）
            self.tokens = 0
            return


async def ask_with_bucket(
    bucket: TokenBucket, question: str, idx: int
) -> str:
    """先拿令牌再发请求"""
    await bucket.acquire()  # 等令牌
    t0 = time.time()
    ans = await ask_async(question)
    print(f"  第{idx:2d}个完成（{time.time()-t0:.2f}s）")
    return ans


async def run_with_token_bucket(rate: float = 3.0) -> float:
    """限制每秒最多发 3 个请求——匀速版"""
    print(f"\n【模式3】Token Bucket 速率={rate}/秒（匀速发出）...")
    bucket = TokenBucket(rate=rate, capacity=int(rate))
    start = time.time()
    results = await asyncio.gather(
        *[ask_with_bucket(bucket, q, i) for i, q in enumerate(QUESTIONS, 1)]
    )
    total = time.time() - start
    print(f"  >>> 耗时 {total:.2f} 秒（最保守，请求匀速发出）")
    return total


# ────────────────────────────────────────────
# 主流程：三种模式跑同样的 10 个请求，对比耗时
# ────────────────────────────────────────────
async def main():
    print("=" * 55)
    print("  进阶第四课：速率限制 + 并发控制")
    print("=" * 55)

    # 模式1：无限并发
    t1 = await run_unlimited()

    # 模式2：Semaphore 限并发
    t2 = await run_with_semaphore(limit=5)

    # 模式3：Token Bucket 限速
    t3 = await run_with_token_bucket(rate=3.0)

    # 三种模式对比
    print("\n" + "=" * 55)
    print(f"  无限并发           ：{t1:.2f} 秒（最快，但可能被限流拒掉）")
    print(f"  Semaphore(5)       ：{t2:.2f} 秒（并发上限，安全）")
    print(f"  TokenBucket 3/秒   ：{t3:.2f} 秒（匀速，最保守）")
    print("=" * 55)
    print("  总结：")
    print("    Semaphore 管'同时几个在飞'——适合控制并发连接数")
    print("    TokenBucket 管'每秒发几个'——适合控制总发送速率")
    print("    实际生产常组合用：Semaphore 防瞬时洪峰 + TokenBucket 防持续过载")


if __name__ == "__main__":
    asyncio.run(main())
