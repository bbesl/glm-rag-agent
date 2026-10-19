"""兼容入口：老命令 python 26_fastapi_agent.py 还能跑

【经验库教训】
不要直接改 26_fastapi_agent.py 的行为——用户习惯了这条命令。
做一层薄包装：import main 然后 uvicorn.run(app)。
新结构跑通后再考虑删除老文件。
"""
import sys
import os

# 关键：把 src/ 加入 sys.path，让 "from customer_service" 能找到包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from customer_service.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
