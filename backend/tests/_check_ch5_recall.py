"""检查 modular 库中对「部门人数」相关查询的召回是否含第五章部门规模表。"""
import asyncio

from app.config import settings
from app.llm.client import create_embeddings
from app.rag.manager import RagManager


def find_ch5(hits):
    for h in hits:
        if "在职人数" in (h.get("text", "") or ""):
            return h.get("text", "")[:120].replace("\n", " | ")
    return None


async def main() -> None:
    embeddings = create_embeddings(fake=False)
    rag = RagManager(settings, embeddings, top_k=settings.rag_top_k, scheme_ids=["modular"])
    scheme = rag.get("modular")
    store = scheme.store
    queries = [
        "张三的部门比李雪的部门哪个人多",
        "张三的部门人数是多少",
        "李雪的部门人数是多少",
        "各部门人员规模与编制",
        "产品部有多少人",
        "各部门人数",
        "各部门在职人数",
        "研发部有多少人",
        "研发部和产品部各有多少人",
        "部门人员编制",
        "各部门规模与编制",
    ]
    for q in queries:
        hits = store.hybrid_search(q, 5)
        ch5 = find_ch5(hits)
        print(f"query={q}")
        for h in hits[:3]:
            print(f"   ({h.get('score', 0):.3f}) {h.get('text', '')[:70].replace(chr(10), ' | ')}")
        print(f"   -> 含第五章：{bool(ch5)}")
        print()


asyncio.run(main())
