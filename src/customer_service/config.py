"""全局配置模块：API 客户端、路径、Redis、日志、常量

【企业级改造关键点】
- BASE_DIR 用 __file__ 计算绝对路径（经验库教训：Docker 内 cwd 不同，
  相对路径会导致 Chroma 数据库文件漂移到意想不到的位置）
- Redis 连接失败自动降级到内存 dict + WARNING 日志（本地开发没装 Redis 也能跑）
- 结构化日志统一在这里配置，各模块用 logging.getLogger(__name__) 拿 logger
"""
import os
import json
import logging
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ── 路径：src/customer_service/config.py → ai-learning/ ──
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/customer_service/
BASE_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))   # ai-learning/ 根目录

# ── .env 加载 ──
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── API 客户端 ──
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


def get_embedding(text: str) -> list[float]:
    """调用智谱 embedding-3 生成 2048 维向量"""
    resp = client.embeddings.create(model="embedding-3", input=text)
    return resp.data[0].embedding


# ── Chroma 向量库（FAQ 持久化） ──
chroma_client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, ".chroma_faq")
)
collection = chroma_client.get_or_create_collection(
    name="faq_customer_service",
    metadata={"hnsw:space": "cosine"},
)
if collection.count() == 0:
    with open(os.path.join(BASE_DIR, "faq_knowledge.txt"), encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
    collection.add(
        documents=chunks,
        embeddings=[get_embedding(c) for c in chunks],
        ids=[f"faq_{i}" for i in range(len(chunks))],
    )


# ════════════════════════════════════════════
# 第 ② 课新增：结构化日志配置
# ════════════════════════════════════════════
# JSON Formatter：让日志能被 ELK/Loki/Grafana 等生产工具解析
class JsonFormatter(logging.Formatter):
    """把 LogRecord 转成 JSON 字符串，生产排障靠它 grep + 聚合"""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> None:
    """配置 root logger，各模块直接 logging.getLogger(__name__)"""
    root = logging.getLogger()
    if root.handlers:
        return  # 已经配过，Docker 多 worker 时防止重复 handler
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


setup_logging()
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════
# 第 ② 课新增：Redis 会话存储（带降级）
# ════════════════════════════════════════════
# 会话过期时间：1 小时无活动自动清除（生产级数据不会无限膨胀）
SESSION_TTL_SECONDS = 3600
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class SessionStore:
    """会话存储抽象层：Redis 可用时用 Redis，不可用时降级到内存 dict

    为什么要抽象类？
    main.py 只调 store.get/set，不关心底下是 Redis 还是 dict。
    将来 PostgreSQL / MongoDB 接进来，加一个实现类就行。
    """
    def get(self, session_id: str) -> list | None:
        raise NotImplementedError

    def set(self, session_id: str, messages: list) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class RedisSessionStore(SessionStore):
    """Redis 实现：JSON 序列化 + TTL 自动过期"""
    def __init__(self, redis_client):
        self.redis = redis_client
        self.prefix = "cs:"  # key 前缀，防止和其他业务数据冲突

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def get(self, session_id: str) -> list | None:
        raw = self.redis.get(self._key(session_id))
        if raw:
            self.redis.expire(self._key(session_id), SESSION_TTL_SECONDS)  # 续期
            return json.loads(raw)
        return None

    def set(self, session_id: str, messages: list) -> None:
        self.redis.setex(
            self._key(session_id), SESSION_TTL_SECONDS,
            json.dumps(messages, ensure_ascii=False)
        )

    def count(self) -> int:
        return len(list(self.redis.scan_iter(f"{self.prefix}*")))


class MemorySessionStore(SessionStore):
    """降级实现：内存 dict（重启丢数据，本地开发用）"""
    def __init__(self):
        self._store: dict[str, list] = {}

    def get(self, session_id: str) -> list | None:
        return self._store.get(session_id)

    def set(self, session_id: str, messages: list) -> None:
        self._store[session_id] = messages

    def count(self) -> int:
        return len(self._store)


# 启动时尝试连 Redis，失败降级（本地没装 Redis 也能跑）
try:
    import redis
    _redis_client = redis.from_url(REDIS_URL, socket_connect_timeout=2)
    _redis_client.ping()  # 连通性测试
    session_store: SessionStore = RedisSessionStore(_redis_client)
    logger.info({"event": "session_store_ready", "backend": "redis", "url": REDIS_URL})
except Exception as e:
    session_store = MemorySessionStore()
    logger.warning({"event": "session_store_fallback", "backend": "memory",
                    "reason": f"Redis 不可用，降级到内存存储: {e}"})


# ════════════════════════════════════════════
# 第 ② 课新增：API Key 鉴权（生产安全基线）
# ════════════════════════════════════════════
# 实际部署时从 .env 读；本地开发可设为空字符串跳过鉴权
API_KEY = os.getenv("API_KEY", "")
