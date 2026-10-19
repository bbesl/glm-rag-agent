# GLM-RAG-Agent：从零手写的 LLM 应用全链路

> 24 课渐进式迭代：从第一次 API 调用，到一套完整的智能客服 Agent 系统。
> 所有核心模块（RAG 检索、Agent 循环、分块器、RRF 融合）均为**手写实现**，再用 LangChain 重构对照——先懂底层，再用框架。

## 项目亮点

**智能客服 Agent**（最终交付物）
- Function Calling 工具编排：模型自主决策调用 FAQ 检索 / 订单查询 / 转人工三类工具
- 多轮对话记忆：跨轮消息累积 + 指代消解（"那个订单的快递单号"→ 自动从历史定位订单号）
- Streamlit Web 界面：可视化工具调用链路，Agent 思考白盒化

**生产级 RAG 全链路**
```
PDF/txt 加载 → 递归分块(+overlap) → 向量化(2048维) → 混合检索 → Reranker 精排 → 带页码溯源生成
                                    ├── 向量检索 (Chroma, 语义)
                                    └── BM25 (精确关键词)
                                          ↓
                                    RRF 倒数排名融合
```

**工程化实践**
| 能力 | 实现 | 实测效果 |
|---|---|---|
| 异步并发 | asyncio + AsyncOpenAI | 5 请求 7.29s → 1.97s（3.7×） |
| 错误韧性 | tenacity 指数退避重试 + 超时控制 + 降级兜底 | 失败用户无感 |
| 可观测性 | Token 用量/耗时/成本全链路日志 | 黑盒变白盒 |
| 速率限制 | Semaphore 并发上限 + TokenBucket 令牌桶 | 快但不超限 |

## 快速开始

```bash
# 1. 克隆 + 创建虚拟环境
git clone https://github.com/<your-name>/glm-rag-agent.git
cd glm-rag-agent
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key（智谱开放平台 https://open.bigmodel.cn 免费申请）
#    在项目根目录创建 .env 文件：
#    ZHIPU_API_KEY=你的key

# 4. 运行
.venv\Scripts\python.exe 13_customer_service_agent.py   # 命令行客服 Agent
.venv\Scripts\python.exe 24_pdf_rag.py                  # PDF 知识库 RAG 问答
streamlit run 15_customer_service_web.py                # Web 界面
```

**技术栈**：Python 3.13 · 智谱 GLM-4-Flash（OpenAI 兼容协议） · Chroma · rank_bm25 · LangChain/LangGraph · tenacity · Streamlit · fpdf2/pypdf

## 24 课迭代路线

| 阶段 | 课 | 主题 | 核心文件 |
|---|---|---|---|
| 基础认知 | 1-2 | LLM 本质（token/无状态）、首次 API 调用 | `01_first_call.py` |
| 核心交互 | 3 | Prompt 工程（system/JSON/Few-shot） | `02_prompt_engineering.py` |
| | 4 | Function Calling 两轮调用 | `03_function_calling.py` |
| RAG | 5 | Embedding 与余弦相似度 | `04_embedding.py` |
| | 6 | 检索流程（切块/排序/top-k） | `05_doc_retrieval.py` |
| | 7 | 完整 RAG 问答（vs 无 RAG 对比） | `06_rag_qa.py` |
| | 8 | Chroma 持久化 + 真实文件读取 | `07_rag_with_chroma.py` |
| | 9 | 交互式问答 | `08_interactive_rag.py` |
| Agent | 10 | Agent 循环（模型自决调几次工具） | `09_first_agent.py` |
| | 11 | 对话记忆（messages 跨轮累积） | `10_agent_memory.py` |
| | 12 | 多 Agent 协作（专员+协调者） | `11_multi_agent.py` |
| 实战项目 | 13 | 客服 RAG（FAQ 知识库） | `12_customer_service_rag.py` |
| | 14 | 客服 Agent（3 工具+记忆） | `13_customer_service_agent.py` |
| | 15 | Streamlit Web 部署 | `15_customer_service_web.py` |
| 工程化 | 16 | 异步并发 3.7× | `16_async_llm.py` |
| | 17 | 重试/超时/降级三道防线 | `17_retry_timeout_fallback.py` |
| | 18 | 可观测性（Token/成本监控） | `18_observability.py` |
| | 19 | Semaphore + TokenBucket 限流 | `19_rate_limit.py` |
| RAG 深水区 | 20 | Reranker 两阶段检索 | `20_reranker.py` |
| | 21 | BM25+向量+RRF 混合检索 | `21_hybrid_search.py` |
| | 22 | 递归分块+overlap | `22_chunking_strategy.py` |
| | 23 | LangChain 重构对照 | `23_langchain_agent.py` |
| | 24 | PDF 解析 + 页码溯源 RAG | `24_pdf_rag.py` |

## 核心踩坑记录（真实调试日志）

1. **tiktoken 毒害第三方 embedding**：`OpenAIEmbeddings` 默认把文本切成 token id 再发请求（OpenAI 私有预处理），智谱 embedding-3 不认 → 检索结果错乱。修复：`check_embedding_ctx_length=False`。
2. **glm-4-flash "假装调工具"**：不加 system 铁律时模型自己编造温度/订单号；LangChain 下还有变种——嘴上说"我需要调工具"却不发 tool_calls。
3. **system prompt 长度悖论**：约束写太长反而稀释工具调用意愿（4 问通过率从 3/4 跌到 1/4），精简版恢复——弱模型的 prompt 工程是做减法。
4. **固定分块切断语义**：150 字硬切 8 块 6 处腰斩句子；递归分块（段落→句号→逗号逐级降级）0 处切断。
5. **tool description 边界**：search_faq 和 check_order 边界重叠导致"发货多久"被误分流到订单查询——description 就是 prompt 工程。

## 设计笔记

- **为什么先手写再用框架**：`create_agent` 内部就是 LangGraph 状态机跑"调用→喂回→循环"，`@tool` 只是 docstring→JSON Schema 的语法糖。手写过一遍，排障时才能分清"框架的锅还是模型的锅"。
- **RRF 为什么用排名不用分数**：向量余弦分数和 BM25 分数量纲完全不同无法直接相加，RRF 只用排名 `1/(k+rank)` 天然归一化。
- **页码溯源**：PDF 分块时把页码写进 metadata，一路带到检索结果和生成 prompt——企业文档问答的可信度靠"（出自第X页）"。

---
*本项目为个人学习项目，24 课全程迭代记录见各文件 docstring。*
