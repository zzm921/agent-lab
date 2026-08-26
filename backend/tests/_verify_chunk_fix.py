"""离线验证 _structure_chunks 修复：第五章部门规模表格块应带「第五章 各部门人员规模与编制」标题。"""
from app.llm.fake_model import FakeEmbeddings
from app.memory.stores.memory_store import MemoryStore
from app.rag.schemes.advanced import AdvancedRagScheme
from app.memory.corpus import KNOWLEDGE_DOCS

vol_text = KNOWLEDGE_DOCS.get("卷十三 组织架构与人员名录", "")
print(f"卷十三字数：{len(vol_text)}")

scheme = AdvancedRagScheme(FakeEmbeddings(), MemoryStore(FakeEmbeddings()))
structured = scheme._structure_chunks(vol_text)
print(f"结构分块数：{len(structured)}")

hits = [t for t, m in structured if "在职人数" in t]
print(f"含在职人数的块：{len(hits)}")
for t in hits:
    print("---")
    print(t[:120].replace("\n", " | "))

# 断言：块文本以第五章标题开头
assert hits, "应存在第五章部门规模块"
assert hits[0].startswith("第五章 各部门人员规模与编制"), "第五章块应带头部章标题"
print("\nOK：第五章块已带标题，可被「部门人数」类查询召回")
