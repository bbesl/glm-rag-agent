# ── 企业级 Dockerfile：FastAPI + 客服 Agent ──
# 经验库教训：Windows 上不能用 gunicorn，用 uvicorn --workers
# 经验库教训：BASE_DIR 用 __file__ 绝对路径，Docker 内 cwd 漂移不会导致 Chroma 库找不到

# 第一阶段：装依赖（缓存层）
FROM python:3.13-slim AS builder
WORKDIR /build

# 只先复制 requirements.txt——依赖不变时 Docker 构建直接命中缓存，巨快
COPY requirements.txt .
# 清华镜像加速（生产环境换成官方 PyPI）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 第二阶段：运行时（比 builder 镜像更小）
FROM python:3.13-slim
WORKDIR /app

# 从 builder 复制已安装的 site-packages（避免重复下载）
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY src/ ./src/
COPY faq_knowledge.txt ./
COPY .env.example ./

# Chroma 数据目录
RUN mkdir -p /app/.chroma_faq

# 环境变量：PYTHONPATH 指向 src/，让 "from customer_service" 能找到包
ENV PYTHONPATH=/app/src
# 经验库教训：.env 用 __file__ 加载，Docker 内 .env 不存在时从环境变量读
ENV ZHIPU_API_KEY=""

EXPOSE 8000

# 启动命令：uvicorn 命令行格式才能开 --workers（不能传 app 对象）
# 生产用 4 workers，测试/单机场景可降为 2
# 注意：不要加 "--access-log true"——它是布尔开关，带值会被当作多余参数报错
CMD ["uvicorn", "customer_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4"]
