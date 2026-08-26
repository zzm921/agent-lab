"""在线验证：修复后「张三的部门比李雪的部门哪个人多」走真实链路（Qdrant + 真实 LLM）。

验证目标：
1. 首轮检索（decompose + multi_recall）命中人员明细但未命中第五章部门规模表；
2. 答案充分性验证判 escalate（而非误判 clarify 需追问澄清）；
3. 升级轮检索命中第五章「各部门人员规模与编制」（研发部130 / 产品部120）；
4. 二次验证可答。

用法：在 backend/ 目录下执行  python tests/_verify_answerability_fix.py
"""
import asyncio

from app.config import settings
from app.llm.client import create_embeddings
from app.rag.manager import RagManager

QUERY = "张三的部门比李雪的部门哪个人多"


def hit_preview(hit: dict, limit: int = 160) -> str:
    text = hit.get("text", "")
    return text[:limit].replace("\n", " | ")


def verdict_summary(verdict: dict) -> str:
    return (
        f"answerable={verdict.get('answerable')} "
        f"recommendation={verdict.get('recommendation')} "
        f"escalate_to={verdict.get('escalate_to')} "
        f"missing={verdict.get('missing_facts')}"
    )


async def main() -> None:
    embeddings = create_embeddings(fake=False)
    rag = RagManager(settings, embeddings, top_k=settings.rag_top_k, scheme_ids=["modular"])
    scheme = rag.get("modular")
    print(f"方案：{scheme.name}  |  集合：{scheme.collection}  |  语料块数：{len(scheme)}\n")

    round_hits: list[list[dict]] = []  # 每轮检索命中
    print(f"===== 查询：{QUERY} =====")
    async for ev in scheme.astream(QUERY, settings.rag_top_k):
        etype = ev["type"]
        if etype == "classify":
            print(
                f"[语义路由] 复杂度={ev.get('complexity')} 检索={ev.get('retrieval_mode')} "
                f"生成={ev.get('generation_mode')} 置信度={ev.get('confidence')} "
                f"理由={ev.get('reason')}"
            )
        elif etype in ("rewrite", "decompose"):
            print(f"[{'改写' if etype == 'rewrite' else '分解'}] {ev.get('rewrites') or ev.get('sub_queries')}")
        elif etype == "multi_hop_plan":
            if ev.get("status") == "done":
                plan = ev.get("plan") or {}
                print(f"[多跳规划] 步骤={[(s['target'], s['query']) for s in plan.get('steps', [])]}")
        elif etype == "multi_hop":
            hop = ev.get("hop") or {}
            print(f"[多跳检索 第{ev.get('index')}跳] query={hop.get('query')} 命中{len(hop.get('hits') or [])}条")
        elif etype == "retrieve":
            hits = ev.get("hits") or []
            round_hits.append(hits)
            print(f"[检索 第{len(round_hits)}轮] 命中{len(hits)}条（reranked={ev.get('reranked')}）：")
            for h in hits:
                print(f"   - ({h.get('score', 0):.3f}) {hit_preview(h)}")
        elif etype == "answerability":
            verdict = ev.get("verdict") or {}
            print(f"[答案充分性 escalated={ev.get('escalated')}] {verdict_summary(verdict)}")
    print("===== 验证要点 =====")
    if len(round_hits) < 2:
        print("!!! 未发生升级检索：答案充分性未判 escalate（或已是最全路径）")
        return
    first, second = round_hits[0], round_hits[1]
    ch5_in_first = any("在职人数" in (h.get("text", "") or "") for h in first)
    ch5_in_second = any("在职人数" in (h.get("text", "") or "") for h in second)
    print(f"首轮命中第五章部门规模表：{ch5_in_first}")
    print(f"升级轮命中第五章部门规模表：{ch5_in_second}")
    if ch5_in_second:
        for h in second:
            text = h.get("text", "")
            if "在职人数" in text:
                for line in text.splitlines():
                    if "研发部" in line or "产品部" in line:
                        print(f"   {line.strip()}")


asyncio.run(main())
