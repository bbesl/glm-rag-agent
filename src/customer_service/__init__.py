"""customer_service 包：企业级客服 Agent API

标准 Python package 结构（src layout），各模块职责单一：
    config.py — 全局配置（API client、路径、常量）
    tools.py  — 三个业务工具（search_faq / check_order / transfer_human）+ tools JSON Schema
    agent.py  — run_agent 循环 + SYSTEM_PROMPT
    main.py   — FastAPI app + 路由定义（对外入口）

运行方式（三种等价）：
    python -m customer_service.main          # 作为包模块运行（推荐，Docker 用这个）
    python ai-learning/src/customer_service/main.py   # 直接跑文件
    老命令 python 26_fastapi_agent.py        # 兼容入口（内部 import main）
"""
from .config import client, BASE_DIR, get_embedding, collection
