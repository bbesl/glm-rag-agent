r"""
第二十四课：PDF 文档解析接入 RAG
=================================
前面 22 课的知识库全是 txt（按空行切段就能用），但真实业务的文档
大多是 PDF——合同、手册、报告。本课打通"PDF → 文本 → 分块 → 入库
→ 带页码溯源的问答"全流程。

【三步主菜】
  ① pypdf 手写解析：PdfReader 逐页 extract_text()——理解 PDF 解析本质
  ② LangChain PyPDFLoader：一行 load()，返回 Document（带页码 metadata）
  ③ RAG 问答：分块（复用第二十二课递归分块）→ 入 Chroma → 检索 → 生成，
     回答标注"出自第几页"——页码 metadata 是 PDF 相对 txt 的核心增值

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\24_pdf_rag.py
"""
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# langchain-community 正在被拆分成独立集成包，import 有 DeprecationWarning
# 目前官方推荐路径仍是它，过滤掉保持输出干净
warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "xyz_handbook.pdf"

load_dotenv(BASE_DIR / ".env")

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from pypdf import PdfReader

# ────────────────────────────────────────────
# 通用：embedding（注意上一课的坑：必须关掉 tiktoken 切分）
# ────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    check_embedding_ctx_length=False,  # 第二十三课的坑：智谱不吃 token id
)

# 递归分块：第二十二课手写的那套，直接搬来用
SEPARATORS = ["\n \n", "\n", "。", "，", ""]  # 从粗到细的分隔符


def recursive_chunk(text: str, chunk_size: int = 200, sep_idx: int = 0) -> list:
    """按分隔符优先级从粗到细切，块超长就换更细的分隔符递归切"""
    if len(text) <= chunk_size or sep_idx >= len(SEPARATORS):
        return [text] if text.strip() else []
    parts = text.split(SEPARATORS[sep_idx])
    if len(parts) == 1:  # 这个分隔符切不动，换更细的
        return recursive_chunk(text, chunk_size, sep_idx + 1)
    chunks, current = [], ""
    for part in parts:
        seg = part + (SEPARATORS[sep_idx] if sep_idx < len(SEPARATORS) - 1 else "")
        if len(seg) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(recursive_chunk(seg, chunk_size, sep_idx + 1))
        elif len(current) + len(seg) > chunk_size:
            chunks.append(current)
            current = seg
        else:
            current += seg
    if current.strip():
        chunks.append(current)
    return chunks


# ════════════════════════════════════════════
# 第一步：pypdf 手写解析——看穿 PDF 解析的本质
# PDF 没有"段落"概念，只有页面上的文字对象，提取=逐页把文字抠出来
# ════════════════════════════════════════════
print("=" * 50)
print("第一步：pypdf 手写解析")
print("=" * 50)
reader = PdfReader(str(PDF_PATH))
print(f"总页数：{len(reader.pages)}")
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    preview = text[:30].replace("\n", " ")
    print(f"  第{i + 1}页 [{len(text)}字] {preview}...")
# 拼成"页文本列表"，第三步 LangChain 版会给出同样的东西
pages_text = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]


# ════════════════════════════════════════════
# 第二步：LangChain PyPDFLoader——一行搞定 + 送你 metadata
# load() 返回 List[Document]，每个 Document = 一页
#   .page_content  页文本   .metadata["page"]  页码（从 0 起）
# 和手写版结果一致，但 Document 结构能直接接 LangChain 全家桶
# ════════════════════════════════════════════
print()
print("=" * 50)
print("第二步：LangChain PyPDFLoader 解析")
print("=" * 50)
docs = PyPDFLoader(str(PDF_PATH)).load()
print(f"共 {len(docs)} 个 Document")
for d in docs[:3]:
    preview = d.page_content[:30].replace("\n", " ")
    print(f"  第{d.metadata['page'] + 1}页 [{len(d.page_content)}字] {preview}...")


# ════════════════════════════════════════════
# 第三步：按页分块 + 页码写进 metadata
# 每页文本（300字左右）超过 chunk_size=200，必须分块（第二十二课递归分块）
# 块的 metadata 带页码 → 检索命中后能溯源"出自第几页"
# ════════════════════════════════════════════
print()
print("=" * 50)
print("第三步：分块（chunk_size=200）+ 页码 metadata")
print("=" * 50)
chunked_docs = []
for d in docs:
    page_no = d.metadata["page"] + 1  # pypdf 页码从 0 起，展示用从 1 起
    for chunk in recursive_chunk(d.page_content, chunk_size=200):
        chunked_docs.append(
            Document(page_content=chunk, metadata={"page": page_no})
        )
print(f"9 页 → {len(chunked_docs)} 块")
for d in chunked_docs[:3]:
    preview = d.page_content[:25].replace("\n", " ")
    print(f"  [第{d.metadata['page']}页] {preview}...")


# ════════════════════════════════════════════
# 第四步：入库新 Chroma 库（count()==0 才入库，第八课模式）
# add_documents 会自动：块文本 → embedding-3 向量 → 存入 collection
# ════════════════════════════════════════════
print()
print("=" * 50)
print("第四步：入库 Chroma（xyz_handbook）")
print("=" * 50)
vectorstore = Chroma(
    collection_name="xyz_handbook",
    persist_directory=str(BASE_DIR / ".chroma_pdf"),
    embedding_function=embeddings,
)
if vectorstore._collection.count() == 0:
    vectorstore.add_documents(chunked_docs)
    print(f"已入库 {len(chunked_docs)} 块（9 页 × 200字/块）")
else:
    print(f"库已存在（{vectorstore._collection.count()} 块），跳过入库直接复用")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# ════════════════════════════════════════════
# 第五步：检索测试——看页码 metadata 的价值
# ════════════════════════════════════════════
print()
print("=" * 50)
print("第五步：检索测试（注意每条结果标注的页码）")
print("=" * 50)
for q in ["年假有几天", "报销超过5000元谁审批", "迟到会怎样"]:
    hits = retriever.invoke(q)
    print(f"\n问：{q}")
    for d in hits:
        preview = d.page_content[:40].replace("\n", " ")
        print(f"  [第{d.metadata['page']}页] {preview}...")


# ════════════════════════════════════════════
# 第六步：完整 RAG 问答——检索结果拼 system，回答标来源页码
# 复用第七课模式：检索 → context → system"禁止编造" → 生成
# ════════════════════════════════════════════
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


def rag_answer(question: str) -> str:
    """检索 top-2 → 拼 system → glm-4-flash 生成带页码溯源的回答"""
    hits = retriever.invoke(question)
    context = "\n\n".join(
        f"【资料（出自第{d.metadata['page']}页）】\n{d.page_content}" for d in hits
    )
    system = (
        "你是 XYZ 公司的人事助手，只根据下面的资料回答问题，"
        "资料里没有的信息就说不知道，禁止编造。\n"
        f"资料：\n{context}\n\n"
        "回答末尾用（出自第X页）标注来源。"
    )
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


print()
print("=" * 50)
print("第六步：完整 RAG 问答（输入 quit 退出）")
print("=" * 50)
while True:
    try:
        q = input("\n你：").strip()
    except EOFError:
        break
    if not q or q.lower() in ("quit", "退出", "q"):
        break
    print("人事助手：", rag_answer(q))
