"""RAG 内嵌知识语料：从 Markdown 语料文件加载（不硬编码在代码里）。

语料以 Markdown 保存于 app/rag/corpus/ 下（云帆科技有限公司行政管理制度汇编.md），
建库（ingest）时读取该文档并按「卷」切分为原始长文本，供各 RAG 方案自行分块入库。
语料为「云帆科技行政管理制度汇编」——强层级结构（卷→章→节→条→表格）的全量制度文本；
其中评测题集卷（含参考答案）不建库，避免带答案的测试题污染检索结果。
"""
from __future__ import annotations

import re
from pathlib import Path

# 语料 Markdown 文件（云帆行政管理制度汇编，按卷组织）
_CORPUS_MD = Path(__file__).resolve().parents[1] / "rag" / "corpus" / "云帆科技有限公司行政管理制度汇编.md"

# 卷标题行：`# 卷…`（H1）
_VOL_RE = re.compile(r"^#\s+(卷.+?)\s*$")
# 含「评测题集」的卷不建库（带参考答案的测试题会污染检索结果）
_EXCLUDE_KEYWORD = "评测题集"


def _parse_corpus() -> dict[str, str]:
    """解析语料 Markdown：按「卷」（H1）切分，每卷保留完整 markdown 原文。

    - 以 `# 卷…` 行为卷边界，卷标题为键、该卷原文为值；
    - 首个 `# 卷` 之前的页眉/目录/说明一律跳过；
    - 卷标题含「评测题集」的卷整卷跳过（卷二十三/四十三/四十八）；
    - `# 卷八·附录`、`# 卷十三·附录` 等按独立卷保留（含 章/节/条/表格 结构）。
    """
    docs: dict[str, str] = {}
    cur_title: str | None = None
    buf: list[str] = []
    for line in _CORPUS_MD.read_text(encoding="utf-8").splitlines():
        m = _VOL_RE.match(line)
        if m:
            if cur_title is not None and buf:
                docs[cur_title] = "\n".join(buf)
            cur_title = m.group(1).strip()
            buf = [line]
            continue
        if cur_title is None:
            continue  # 卷标题前的页眉/目录/说明，不入库
        buf.append(line)
    if cur_title is not None and buf:
        docs[cur_title] = "\n".join(buf)
    return {k: v for k, v in docs.items() if _EXCLUDE_KEYWORD not in k}


# 分卷文档：{卷标题: 卷原文}（供结构感知分块/后续 Graph RAG 按卷建图）
KNOWLEDGE_DOCS: dict[str, str] = _parse_corpus()

# 平坦语料：每条为「一卷」的完整原文（保留 章/节/条/表格 结构），由各方案自行分块
KNOWLEDGE_CORPUS: list[str] = list(KNOWLEDGE_DOCS.values())
