# 部署上线指南

> 三种方案从简到繁，**推荐 Docker Compose**（一键启动、跨平台、环境一致）。
> 如果你用 Linux 服务器且不想装 Docker，用方案 ②。Windows 服务器用方案 ③。

---

## 方案 ①：Docker Compose（推荐）

### 前置条件

1. 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)（Windows/Mac）或 Docker Engine（Linux）
2. 确认 Docker 可用：`docker --version` 和 `docker compose version` 都不报错
3. **国内网络必做**：配置镜像加速器（否则拉 `python:3.13-slim`/`redis:7-alpine` 会超时）——见文末「附录：国内镜像加速器」

### 部署步骤

```bash
# 1. 进入项目目录
cd ai-learning

# 2. 创建 .env（如果还没有）
cp .env.example .env
# 然后编辑 .env，填入你的 ZHIPU_API_KEY
# 可选：填 API_KEY="your-secret-key" 启用鉴权

# 3. 构建镜像 + 启动（后台运行）
docker compose up -d --build

# 4. 查看日志
docker compose logs -f api      # 只看 api 服务
docker compose logs -f          # 全部服务

# 5. 健康检查（宿主机端口是 8080，容器内仍是 8000）
curl http://localhost:8080/health
# 返回 {"status":"ok","sessions":1} 即成功
```

> **端口说明**：host 端口用 **8080** 映射到容器内 8000。原因：Windows 上 8000 可能落在 Hyper-V 保留端口段（`netsh interface ipv4 show excludedportrange protocol=tcp` 可查），导致 `bind: access permissions` 报错。Linux 上可直接改成 `"8000:8000"`。

### 常用命令

```bash
docker compose up -d --build    # 构建并启动（代码改了重跑这个）
docker compose down             # 停止并删除容器（数据卷保留）
docker compose down -v          # 彻底删除（包括 Chroma 和 Redis 数据）
docker compose restart api      # 只重启 api 服务
docker compose logs -f api      # 实时追 api 日志
docker compose exec api python -c "import customer_service"  # 进容器测试
```

### 数据持久化

Docker Compose volumes 自动处理：
| 数据 | Volume | 容器删了 |
|---|---|---|
| Chroma FAQ 向量库 | `chroma_data` | ✅ 保留 |
| Redis 会话 | `redis_data` | ✅ 保留 |

想彻底清空重新来过：`docker compose down -v`

---

## 方案 ②：Linux 原生部署（Ubuntu 22.04+）

适合不想装 Docker 的场景，用 systemd 管理进程 + Nginx 反向代理。

### 前置条件

- Python 3.11+、Redis 7+、Nginx
- 服务器开放 80 端口（Nginx 用），8000 端口只监听 127.0.0.1（Nginx 转发）

### 步骤

```bash
# 1. 装依赖
sudo apt update
sudo apt install python3 python3-venv python3-pip redis-server nginx

# 2. 部署代码
cd /opt
git clone <your-repo-url> ai-learning
cd ai-learning
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. 启动 Redis（如果没装）
sudo systemctl enable redis-server
sudo systemctl start redis-server

# 4. 创建 .env
cp .env.example .env
nano .env  # 填入 ZHIPU_API_KEY 和可选 API_KEY

# 5. systemd 服务文件（让 uvicorn 开机自启 + 崩溃自动重启）
sudo tee /etc/systemd/system/cs-api.service << 'EOF'
[Unit]
Description=Customer Service Agent API
After=network.target redis-server.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ai-learning
Environment=PYTHONPATH=/opt/ai-learning/src
EnvironmentFile=/opt/ai-learning/.env
ExecStart=/opt/ai-learning/.venv/bin/uvicorn \
    customer_service.main:app \
    --host 127.0.0.1 --port 8000 \
    --workers 4 --log-level info
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# 6. 启动服务
sudo systemctl daemon-reload
sudo systemctl enable cs-api
sudo systemctl start cs-api
sudo systemctl status cs-api    # 确认 active (running)

# 7. Nginx 反向代理
sudo tee /etc/nginx/sites-available/cs-api << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # LLM 调用耗时较长，调大超时
        proxy_read_timeout 120s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/cs-api /etc/nginx/sites-enabled/
sudo nginx -t                    # 配置检查
sudo systemctl reload nginx      # 重载

# 8. 验证
curl http://localhost:8000/health   # 直连 uvicorn
curl http://your-domain.com/health  # 走 Nginx
```

### 维护命令

```bash
sudo systemctl status cs-api     # 查看服务状态
sudo journalctl -u cs-api -f     # 实时看日志
sudo systemctl restart cs-api    # 重启（代码更新后）
```

---

## 方案 ③：Windows 原生部署（NSSM）

适合 Windows Server 场景，用 NSSM 把 uvicorn 注册为 Windows 服务（开机自启 + 崩溃自动重启）。

### 前置条件

- 安装 [Redis for Windows](https://github.com/tporadowski/redis/releases)（或用 WSL 跑 Redis）
- 下载 [NSSM](https://nssm.cc/download)（Non-Sucking Service Manager）
- Python 3.11+

### 步骤

```powershell
# 1. 部署代码 + 装依赖
cd C:\apps\ai-learning
git clone <your-repo-url> .
.venv\Scripts\pip.exe install -r requirements.txt

# 2. 启动 Redis（如果没装）
# 下载 Redis for Windows 后：
# redis-server.exe --service-install --service-name Redis
# redis-server.exe --service-start

# 3. 创建 .env
copy .env.example .env
# 编辑 .env 填入 ZHIPU_API_KEY

# 4. 用 NSSM 注册 uvicorn 为服务
# nssm.exe 在 NSSM 下载包里，放到 PATH 或指定完整路径
nssm install CsApi "C:\apps\ai-learning\.venv\Scripts\python.exe" "-m uvicorn customer_service.main:app --host 0.0.0.0 --port 8000 --workers 4"

# 设置工作目录（关键！否则 __file__ 绝对路径计算错误）
nssm set CsApi AppDirectory "C:\apps\ai-learning"

# 设置环境变量（可选，.env 已有 load_dotenv）
nssm set CsApi AppEnvironment PYTHONPATH=C:\apps\ai-learning\src

# 设置依赖（Redis 先启动）
nssm set CsApi DependOnService Redis

# 自动重启
nssm set CsApi AppStdout "C:\apps\ai-learning\logs\stdout.log"
nssm set CsApi AppStderr "C:\apps\ai-learning\logs\stderr.log"

# 启动
nssm start CsApi

# 验证
Invoke-RestMethod http://localhost:8000/health
```

---

## API 使用说明

### 健康检查（不鉴权）
```bash
curl http://localhost:8080/health
# {"status":"ok","sessions":1}
```

### 聊天（鉴权）
```bash
# 无 API_KEY（开发模式，.env 里留空）
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user-001","message":"发货多久到"}'

# 有 API_KEY（生产模式）
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"session_id":"user-001","message":"订单100001到哪了"}'
```

### 交互文档（FastAPI 自动生成）
部署后浏览器打开：`http://localhost:8080/docs`

---

## 故障排查

| 问题 | 排查方法 |
|---|---|
| Redis 连接失败 | Docker：`docker compose exec api python -c "import redis; r=redis.from_url('redis://redis:6379/0'); print(r.ping())"` 本地：`redis-cli ping` |
| Chroma 找不到 | 检查 .chroma_faq/ 目录是否在 BASE_DIR 下（BASE_DIR = 项目根目录） |
| API Key 鉴权失败 | 确认 .env 里 API_KEY 和请求头 X-API-Key 一致；空字符串则开发模式放行 |
| 端口被占用 | Linux：`lsof -i :8000`；Windows：`netstat -ano \| findstr :8000` 然后 `taskkill /PID xxx /F` |
| Docker build 失败 | 检查 .dockerignore 是否把 .env 排除；requirements.txt 是否完整 |

---

## 架构图

```
                    ┌─────────────┐
   用户请求 ────────▶│   Nginx     │
                    │  (80 端口)   │
                    └──────┬──────┘
                           │ 反向代理
                    ┌──────▼──────┐
                    │  FastAPI    │
                    │ :8000       │
                    │             │
                    │  ┌────────┐ │
                    │  │ Redis  │ │  会话存储
                    │  │ :6379  │ │  (TTL 1h 自动过期)
                    │  └────────┘ │
                    │             │
                    │  ┌────────┐ │
                    │  │ Chroma │ │  FAQ 向量库
                    │  │        │ │  (持久化卷)
                    │  └────────┘ │
                    │             │
                    │  ┌────────┐ │
                    │  │ LLM    │ │  智谱 GLM-4-Flash
                    │  └────────┘ │
                    └─────────────┘
```

---

## 附录：国内镜像加速器

Docker Hub（`registry-1.docker.io`）在国内访问不稳定，拉镜像会超时。需要给 Docker daemon 配镜像加速器。

### Windows（Docker Desktop）

Settings → Docker Engine，在 JSON 里加 `registry-mirrors`，点 Apply & Restart。配置文件实际位置：`C:\Users\<你>\.docker\daemon.json`

```json
{
  "builder": { "gc": { "defaultKeepStorage": "20GB", "enabled": true } },
  "experimental": false,
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1panel.live",
    "https://docker.1ms.run",
    "https://dockerproxy.net"
  ]
}
```

命令行重启：`docker desktop restart`。验证：`docker info --format '{{json .RegistryConfig.Mirrors}}'`

### Linux（Docker Engine）

编辑 `/etc/docker/daemon.json`（同上 JSON），然后：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 备注

- 镜像源会失效/更换，若全部不通，用 `Invoke-WebRequest "$mirror/v2/"` 测连通性（返回 200 或 401 = 可用），或改用阿里云个人加速器（需登录阿里云容器镜像服务获取专属地址）。
- 这些加速器只加速**拉取**；`pip install` 走的是 requirements 里已配的清华源。
