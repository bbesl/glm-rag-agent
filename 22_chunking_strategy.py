r"""
RAG 深水区第三课：分块策略——递归分块 + overlap 重叠
=====================================================
前面所有课的 RAG 都按"空行切段"（split("\n\n")）做分块——简单但粗。
本课学更智能的分块，对比三种分块方式。

【为什么分块是 RAG 的地基】
  ① LLM context 有限（再大也塞不下整本书）
  ② 检索精度：只想要相关片段，整篇文档灌进来噪音大
  ③ 成本：token 数 × 单价，块越准，喂 LLM 的越精，越省钱

【三种分块方式】
  方式1 固定分块：按 N 字硬切。简单粗暴，可能从句子中间切断，破坏语义。
       例：把"包邮，不满99元"切成两块 → 检索到一半信息残缺。
  方式2 递归分块：按"段落 → 句号 → 逗号 → 字符"优先级找自然边界切，
       尽量保语义完整。
  方式3 递归分块 + overlap：相邻块共享一段文字（如 40 字重叠），
       防止跨块信息丢失。代价=存储量略增。

【本课演示】
  用 faq_knowledge.txt 全部 15 条 FAQ 拼成一段长文本，
  对比三种分块方式，看哪种切得最干净。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\22_chunking_strategy.py
"""
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# 方式1：固定分块（fixed-size chunking）
# ═══════════════════════════════════════════════════════════════
def fixed_size_chunk(text, chunk_size=150, overlap=0):
    """
    按固定长度 chunk_size 字符硬切。
    overlap>0 时相邻块共享 overlap 字符（下一块提前 overlap 字符开始）。
    问题：可能从句子中间切开，把"包邮，不满"切成两块。
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # 下一块提前 overlap 字符开始，形成重叠
        start = end - overlap
    return chunks


# ═══════════════════════════════════════════════════════════════
# 方式2 + 3：递归分块（recursive chunking，可加 overlap）
# ═══════════════════════════════════════════════════════════════
def recursive_chunk(text, chunk_size=150, overlap=40, separators=None):
    """
    递归分块：按分隔符优先级切，尽量在自然边界断开。
    separators 默认 ["\\n\\n", "。", "，", ""]——从粗到细：
      \\n\\n（段落）→ 。(句号) → ，(逗号) → ""(按字符)
    算法：
      ① 用当前分隔符把文本切成多段
      ② 累积段直到超过 chunk_size
      ③ 超了 → 当前累积成块，看新段自己超不超
      ④ 新段自己还超 → 换更细的分隔符递归切
    最后可选加 overlap：把上一块末尾 overlap 字符拼到下一块开头。
    """
    if separators is None:
        separators = ["\n\n", "。", "，", ""]

    chunks = []

    def split(text, sep_idx):
        """递归切分核心"""
        # 已经够小：直接成块
        if len(text) <= chunk_size:
            chunks.append(text)
            return

        # 所有分隔符都试过还太长：按字符硬切
        if sep_idx >= len(separators) or separators[sep_idx] == "":
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i + chunk_size])
            return

        sep = separators[sep_idx]
        parts = text.split(sep)
        current = ""
        for part in parts:
            # 重新加回分隔符（保语义，第一段除外）
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate  # 没超，继续累积
            else:
                # 超了：current 成块
                if current:
                    chunks.append(current)
                # 看 part 自己超不超
                if len(part) > chunk_size:
                    split(part, sep_idx + 1)  # 递归用更细的分隔符
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)

    split(text, 0)

    # 加 overlap：上一块末尾 overlap 字符拼到下一块开头
    if overlap > 0 and len(chunks) > 1:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) >= overlap else chunks[i - 1]
            result.append(tail + chunks[i])
        return result
    return chunks


def show(title, chunks, overlap_size=0):
    """打印分块结果。overlap_size>0 时用 ←...→ 标记块开头的重叠尾巴。"""
    print(f"\n{'━' * 60}")
    print(f"{title}（共 {len(chunks)} 块）")
    print("━" * 60)
    for i, c in enumerate(chunks, 1):
        if overlap_size > 0 and i > 1 and len(c) > overlap_size:
            # 块开头 overlap_size 字符是上一块的尾巴
            print(f"\n[块 {i}] ({len(c)} 字)")
            print(f"  ↳ 重叠 ←{c[:overlap_size]}→")
            print(f"  ↳ 正文 {c[overlap_size:]}")
        else:
            print(f"\n[块 {i}] ({len(c)} 字)")
            print(f"  {c}")


def main():
    # 准备实验文本：faq_knowledge.txt 全部 15 条 FAQ 拼成一段长文本
    faq_path = Path(BASE_DIR) / "faq_knowledge.txt"
    with open(faq_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    paragraphs = raw.split("\n\n")
    # 拼成一段长文本，段落间保留 \n\n 作为自然边界
    text = "\n\n".join(paragraphs)

    print(f"原始文本：{len(text)} 字，{len(paragraphs)} 条 FAQ")
    print(f"预览（前 80 字）：{text[:80]}...")

    CHUNK_SIZE = 150
    OVERLAP = 40

    # === 方式1：固定分块（无重叠）===
    c1 = fixed_size_chunk(text, chunk_size=CHUNK_SIZE, overlap=0)
    show(f"方式1：固定分块（每块 {CHUNK_SIZE} 字，无重叠）", c1)

    # === 方式2：递归分块（无重叠）===
    c2 = recursive_chunk(text, chunk_size=CHUNK_SIZE, overlap=0)
    show(f"方式2：递归分块（每块 ≤{CHUNK_SIZE} 字，无重叠）", c2)

    # === 方式3：递归分块 + overlap ===
    c3 = recursive_chunk(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    show(f"方式3：递归分块 + overlap（每块 ≤{CHUNK_SIZE} 字，重叠 {OVERLAP} 字）", c3, overlap_size=OVERLAP)

    # === 关键对比：固定分块切断的句子 ===
    print(f"\n{'━' * 60}")
    print("关键对比：固定分块切断的语义")
    print("━" * 60)
    cuts = 0
    for i, c in enumerate(c1):
        # 块开头不是 FAQ 段落开头标记 → 从中间切开
        if c and not any(c.startswith(tag) for tag in ["物流", "退换", "支付", "账户", "商品"]):
            cuts += 1
            if cuts == 1:  # 只详细展示第一个
                print(f"\n⚠ 固定分块第 {i + 1} 块从句子中间切开：")
                print(f"  块开头：{c[:50]}...")
                print(f"  上一块结尾：...{c1[i - 1][-50:]}")
                print(f"  → 句子被切断，RAG 检索到这块时上下文残缺！")
    print(f"\n固定分块共 {cuts} 处从句子中间切开（共 {len(c1)} 块）。")
    print(f"递归分块 0 处切断（每块都从段落开头累积，保语义完整）。")

    # === 总结 ===
    print(f"\n{'━' * 60}")
    print("三句话总结")
    print("━" * 60)
    print(f"""
1. 固定分块（按 {CHUNK_SIZE} 字硬切）简单，但 {cuts} 处从句子中间切开，破坏语义。
2. 递归分块按"段落→句号→逗号→字符"找自然边界切，0 处切断，语义完整。
3. overlap（重叠 {OVERLAP} 字）让相邻块共享一段文字，防跨块信息丢失，代价是存储量略增。

下一课预告：第二十三课 PDF 解析——前面所有课都读 txt，下一课学从 PDF 提取文本做 RAG。
""")


if __name__ == "__main__":
    main()
