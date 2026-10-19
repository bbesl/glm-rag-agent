"""
第十八课：可观测性——让 LLM 调用白盒化
============================================================
前三课回顾：
- 第16课：异步并发让请求快了（7.29s -> 1.97s）
- 第17课：重试+超时+降级让应用不崩（三道防线）
- 第18课：可观测性让你知道它怎么跑的、崩在哪了、花了多少钱

可观测性（Observability）三维度：
1. token 用量——每次调用用了多少 token（prompt + completion）
2. 调用链路——每次调用的细节：耗时、状态、错误
3. 成本核算——根据 token 用量和模型定价算花了多少钱

实现方式：自建轻量 LLMCallLogger（不引 Langfuse，零额外依赖）
"""
import os
import time
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# ====== 环境加载 ======
load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ====== 模型定价表（元/百万 token，仅作教学示例，实际以官方为准）======
PRICE_TABLE = {
    "glm-4-flash": {"input": 0.0,  "output": 0.0},   # 免费
    "glm-4-air":   {"input": 1.0,  "output": 1.0},    # 1元/百万token
    "glm-4":       {"input": 50.0, "output": 50.0},  # 50元/百万token
}


class LLMCallLogger:
    """LLM 调用日志记录器——可观测性的地基

    每次调用记一条，最后能：
    - 看明细（哪次调用花了多少 token、多久、成没成功）
    - 看汇总（总共花了多少 token、多少钱、平均多久）
    """

    def __init__(self, model="glm-4-flash"):
        self.model = model
        self.logs = []              # 每条记录是一个 dict
        self.start_time = time.time()  # 程序启动时间（算墙钟时间）

    def log_call(self, messages, response, duration, status="success", error=None):
        """记录一次 LLM 调用"""
        # 从 response.usage 提取 token 用量（这是可观测性的核心数据源）
        usage = getattr(response, "usage", None) if response else None
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        # 成本核算：token 数 × 单价 / 1,000,000（因为定价是"元/百万token"）
        price = PRICE_TABLE.get(self.model, {"input": 0, "output": 0})
        cost = (prompt_tokens * price["input"]
                + completion_tokens * price["output"]) / 1_000_000

        entry = {
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "model": self.model,
            "status": status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_s": round(duration, 3),
            "cost_yuan": round(cost, 6),
            "error": error,
            "prompt_preview": (messages[-1].get("content", "")[:50] if messages else ""),
            "response_preview": (
                response.choices[0].message.content[:50]
                if status == "success" and response and response.choices else ""
            ),
        }
        self.logs.append(entry)
        return entry

    def summary(self):
        """汇总统计——把所有调用加总"""
        total_calls = len(self.logs)
        success = sum(1 for l in self.logs if l["status"] == "success")
        total_prompt = sum(l["prompt_tokens"] for l in self.logs)
        total_completion = sum(l["completion_tokens"] for l in self.logs)
        total_cost = sum(l["cost_yuan"] for l in self.logs)
        sum_duration = sum(l["duration_s"] for l in self.logs)

        return {
            "total_calls": total_calls,
            "success": success,
            "failed": total_calls - success,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_yuan": round(total_cost, 6),
            "sum_duration_s": round(sum_duration, 3),
            "wall_time_s": round(time.time() - self.start_time, 3),
        }


def ask_llm(client, logger, messages, model="glm-4-flash"):
    """带可观测性的 LLM 调用——每次调用都记日志

    和普通调用唯一的区别：包了计时 + try/except + logger.log_call
    """
    start = time.time()
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        duration = time.time() - start
        logger.log_call(messages, response, duration, status="success")
        return response.choices[0].message.content
    except Exception as e:
        duration = time.time() - start
        logger.log_call(messages, None, duration, status="failed", error=str(e))
        raise


def main():
    logger = LLMCallLogger(model="glm-4-flash")

    print("=" * 60)
    print("第十八课：可观测性——让 LLM 调用白盒化")
    print("=" * 60)

    # 三次调用（模拟一个真实应用里连续问几个问题）
    calls = [
        ([
            {"role": "system", "content": "你是个简洁的助手，回答不超过20字。"},
            {"role": "user", "content": "用一句话解释什么是 token。"},
        ]),
        ([{"role": "user", "content": "Python 的 GIL 是什么？一句话。"}]),
        ([{"role": "user", "content": "异步和并行的区别？一句话。"}]),
    ]

    for i, msgs in enumerate(calls, 1):
        q = msgs[-1]["content"]
        print(f"\n[调用{i}] {q}")
        answer = ask_llm(client, logger, msgs)
        print(f"回答: {answer}")

    # ====== 调用明细日志 ======
    print("\n" + "=" * 60)
    print("调用明细日志（可观测性：看每次调用细节）")
    print("=" * 60)
    for i, log in enumerate(logger.logs, 1):
        print(f"\n--- 调用 #{i} ---")
        print(f"时间 {log['time']} | 状态 {log['status']} | 耗时 {log['duration_s']}s")
        print(f"token: prompt={log['prompt_tokens']} completion={log['completion_tokens']} total={log['total_tokens']}")
        print(f"成本 ¥{log['cost_yuan']} | 问题: {log['prompt_preview']}")
        if log["response_preview"]:
            print(f"回答: {log['response_preview']}")
        if log["error"]:
            print(f"错误: {log['error']}")

    # ====== 汇总统计 ======
    print("\n" + "=" * 60)
    print("汇总统计（可观测性：看整体开销）")
    print("=" * 60)
    s = logger.summary()
    print(f"调用次数: {s['total_calls']} (成功 {s['success']} 失败 {s['failed']})")
    print(f"总 token: {s['total_tokens']} (prompt {s['total_prompt_tokens']} + completion {s['total_completion_tokens']})")
    print(f"总成本: ¥{s['total_cost_yuan']}")
    print(f"累计耗时 {s['sum_duration_s']}s | 程序运行 {s['wall_time_s']}s")

    # 成本对比：如果换成收费模型 glm-4
    glm4_cost = (s['total_prompt_tokens'] * 50
                 + s['total_completion_tokens'] * 50) / 1_000_000
    print(f"\n成本对比: 同样这些调用换 glm-4 要花 ¥{round(glm4_cost, 4)}（flash 免费）")

    # ====== 写日志文件持久化 ======
    log_file = "ai-learning/llm_calls.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"# LLM 调用日志 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for i, log in enumerate(logger.logs, 1):
            f.write(f"--- 调用 #{i} ---\n")
            f.write(json.dumps(log, ensure_ascii=False, indent=2) + "\n\n")
        f.write("--- 汇总 ---\n")
        f.write(json.dumps(s, ensure_ascii=False, indent=2) + "\n")
    print(f"\n日志已写入: {log_file}")


if __name__ == "__main__":
    main()
