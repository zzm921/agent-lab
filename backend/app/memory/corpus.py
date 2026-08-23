"""RAG 内嵌知识语料：从 Markdown 语料文件加载（不硬编码在代码里）。

语料以 Markdown 保存于 app/rag/ 下（五代RAG梯度测试统一语料*.md），
建库（ingest）时读取该文档并解析为「分章节的原始长文本」，供各 RAG 方案自行分块入库。
语料为虚构「科创公司员工行政、考勤、福利与差旅全管理制度」——长文本、高冗余、
规则嵌套、信息碎片化，天然制造「固定切块切断语义」「跨块信息无法关联」「多条件叠加」
等检索难题，为 Naive 之后 Advanced / Modular / Graph / Agentic 逐代升级提供同源语料。
"""
from __future__ import annotations

from pathlib import Path

# 语料 Markdown 文件（与五代 RAG 迭代演示文档同目录）
_CORPUS_MD = Path(__file__).resolve().parents[1] / "rag" / "五代RAG梯度测试统一语料（适配分块策略+全版本RAG迭代演示）.md"


def _parse_corpus() -> dict[str, list[str]]:
    """解析语料 Markdown：按「### 章节」分组的段落文本。

    - 以 `###` 行作为章节边界，章节名去「### 」前缀；
    - 跳过 `#`/`##` 标题行、`>` 引用行（注记）与空行；
    - 其余每个非空行视为一个段落（单行长文本），作为原始语料；
    - 对文本做 Markdown 反转义（`\\.`/`\\-`），使检索与展示更自然。
    """

    def _unescape(text: str) -> str:
        return text.replace("\\.", ".").replace("\\-", "-")

    docs: dict[str, list[str]] = {}
    section = "总则"
    for line in _CORPUS_MD.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("###"):
            section = stripped.lstrip("#").strip()
            docs.setdefault(section, [])
            continue
        if stripped.startswith("#") or stripped.startswith(">"):
            continue  # 说明性标题/注记，不入库
        docs.setdefault(section, []).append(_unescape(stripped))
    return docs


# 分文档分组（供后续 Graph RAG 按文档维度建图；当前入库使用平坦语料）
KNOWLEDGE_DOCS: dict[str, list[str]] = _parse_corpus()

# 平坦语料：当前 ingest_all 的输入（每条为一段长文本，由各方案自行分块）
KNOWLEDGE_CORPUS: list[str] = [paragraph for paragraphs in KNOWLEDGE_DOCS.values() for paragraph in paragraphs]
