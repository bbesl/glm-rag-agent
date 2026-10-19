"""FastAPI 应用入口：路由定义 + 会话管理 + 日志 + 鉴权

【模块边界】
- 这里只关心"对外接口是什么"：请求/响应模型 + 路由函数
- Agent 循环在 agent.py，工具在 tools.py，配置/Redis/日志/鉴权都在 config.py
- 将来换 DB、换鉴权方式，只改 import + 路由内部调用

三种启动方式（等价）：
    python -m customer_service.main                    # Docker 用这个
    python src/customer_service/main.py                # 直接跑文件
    python run.py                                       # 兼容入口
"""
import time
import logging
import uvicorn
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .agent import run_agent, SYSTEM_PROMPT
from .config import session_store, API_KEY, logger


# ── Pydantic 模型 ──
class ChatRequest(BaseModel):
    session_id: str = Field(..., examples=["user-9527"], description="会话ID，同一个ID共享对话记忆")
    message: str = Field(..., min_length=1, examples=["订单100001到哪了"], description="用户消息")


class ChatReply(BaseModel):
    reply: str = Field(..., description="客服回答")
    tool_log: list = Field(..., description="本轮工具调用链路")


# ── FastAPI app ──
app = FastAPI(
    title="好物优选智能客服 API",
    description="企业级客服 Agent（第 ② 课：Redis会话+结构化日志+API鉴权）",
    version="2.0.0",
)


# ════════════════════════════════════════════
# 第 ② 课新增 1：请求日志中间件
# ════════════════════════════════════════════
# 生产级可观测性：每个请求记 method/path/status/duration_ms
# 结构化 JSON 日志，Grafana/Loki 可直接做 dashboard
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    status = 500  # 默认 500：call_next 抛非 HTTPException 时 finally 也能取到值（防 UnboundLocalError）
    try:
        response = await call_next(request)
        status = response.status_code
    except HTTPException as exc:
        status = exc.status_code
        raise
    except Exception:
        status = 500
        raise
    finally:
        duration_ms = round((time.time() - start) * 1000, 2)
        logger.info({
            "event": "http_request",
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": duration_ms,
        })
    return response


# ════════════════════════════════════════════
# 第 ② 课新增 2：API Key 鉴权（FastAPI Dependency）
# ════════════════════════════════════════════
# Dependency 能自动在 /docs 里显示锁图标，且可按路由粒度控制
# API_KEY 为空字符串时跳过鉴权（本地开发便利）
def verify_api_key(x_api_key: str | None = Header(default=None, description="API Key")):
    """API Key 鉴权：API_KEY 为空放行（本地开发），否则必须匹配"""
    if not API_KEY:
        return  # 本地开发：API_KEY 为空时放行
    if not x_api_key:
        logger.warning({"event": "auth_failed", "reason": "missing_api_key"})
        raise HTTPException(status_code=401, detail="API Key required")
    if x_api_key != API_KEY:
        logger.warning({"event": "auth_failed", "reason": "invalid_api_key"})
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


# ── 路由 ──
@app.get("/health")
def health():
    """健康检查：生产环境负载均衡/容器探活靠它（不鉴权）"""
    return {"status": "ok", "sessions": session_store.count()}


@app.post("/chat", response_model=ChatReply, dependencies=[Depends(verify_api_key)])
def chat(req: ChatRequest) -> ChatReply:
    # 第 ② 课改造：用 Redis/Memory SessionStore 替换 dict
    messages = session_store.get(req.session_id)
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": req.message})

    reply, tool_log = run_agent(messages)

    # 写回 Redis（带 TTL 自动过期）
    session_store.set(req.session_id, messages)

    # 结构化日志：记录对话摘要（不打完整 messages，避免日志爆炸 + 隐私泄露）
    logger.info({
        "event": "chat_completed",
        "session_id": req.session_id,
        "tool_count": len(tool_log),
        "turns": len(tool_log) + 1,
        "reply_preview": reply[:50],
    })

    return ChatReply(reply=reply, tool_log=tool_log)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
