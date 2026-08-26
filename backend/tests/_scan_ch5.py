"""扫描 modular 集合全部块，确认第五章「各部门人员规模与编制」是否入库。"""
from app.config import settings
from app.llm.client import create_embeddings
from app.rag.manager import RagManager

embeddings = create_embeddings(fake=False)
rag = RagManager(settings, embeddings, top_k=settings.rag_top_k, scheme_ids=["modular"])
scheme = rag.get("modular")
store = scheme.store

# 直接访问底层 Qdrant 客户端扫描全部点
qdrant = None
for b in (store.backends if hasattr(store, "backends") else [store]):
    if b.__class__.__name__ == "QdrantStore":
        qdrant = b
        break
assert qdrant is not None, "未找到 QdrantStore 后端"
client = qdrant.client
col = qdrant.collection
offset = None
found = []
total = 0
while True:
    res = client.scroll(collection_name=col, limit=200, offset=offset, with_payload=True, with_vectors=False)
    points, next_offset = res
    if not points:
        break
    total += len(points)
    for p in points:
        text = (p.payload or {}).get("text", "") or ""
        if "在职人数" in text or "各部门人员规模" in text:
            found.append((p.id, p.payload.get("章", ""), text[:100].replace("\n", " | ")))
    offset = next_offset
    if next_offset is None:
        break

print(f"扫描块数：{total}")
print(f"含第五章/在职人数的块：{len(found)}")
for fid, chap, prev in found[:10]:
    print(f"  id={fid} 章={chap}  {prev}")
    p = client.retrieve(collection_name=col, ids=[fid], with_payload=True)[0]
    print(f"  payload keys={list((p.payload or {}).keys())}")
    print(f"  metadata={p.payload.get('metadata')}")
