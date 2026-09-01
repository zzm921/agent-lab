"""增量入库测试：台账判重 → 只重算变更文档；删除/淘汰/容器移除的旧块清理。

全链路走真实 run_pipeline（解析+清洗）+ run_incremental_ingest（台账增量），
存储用内存后端 + FakeEmbeddings，不联网、不依赖 Key。
"""
import io
import zipfile
from pathlib import Path

from app.llm.fake_model import FakeEmbeddings
from app.rag.manager import RagManager
from app.rag.preprocess import run_pipeline
from app.rag.preprocess.ledger import LedgerStore
from scripts.ingest_documents import run_incremental_ingest
from tests.conftest import make_settings

POLICY_TEXT = (
    "项目管理制度总则规定，全员应当遵守研发与文档各项规范安排。"
    "研发文档须在评审通过后归档至知识库，注明作者、版本与生效日期。"
    "版本变更需在变更记录中写明修改原因与影响范围，经负责人审批后生效。"
    "知识库文档每季度复审一次，过期文档标记为历史版本并从主索引移除。"
    "本制度自发布之日起施行，由项目管理办公室负责解释与修订完善。"
)
TRAVEL_TEXT = (
    "差旅报销管理规定明确，员工因公出差须事先提交申请并获部门负责人批准。"
    "交通工具选择以经济实用为原则，同城高铁两小时以内原则上不安排飞机出行。"
    "住宿费用按职级分档报销，票据须为本人实名且日期与行程一致方可受理。"
    "报销材料应当在返回后十个工作日内提交，逾期未提交的视为自动放弃处理。"
    "财务部按月抽查报销凭证，发现虚报冒领的移交人力资源部按制度处理。"
)
MEETING_TEXT = (
    "会议室使用管理规定要求，各部门使用会议室须提前一天预约登记。"
    "单场会议时长原则上不超过两小时，超时须现场确认后续时段是否空闲。"
    "会后使用人应当清理白板、归位桌椅并关闭投影与空调等设备电源。"
    "外部来访人员使用会议室由接待部门统一申请，并负责全程陪同管理。"
    "行政部每周汇总会议室使用情况，对多次违约使用的部门进行通报。"
)


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _build_rag(tmp_path) -> tuple[RagManager, LedgerStore]:
    settings = make_settings(rag_schemes=["naive"])  # 未配 Qdrant/ES → 内存后端
    rag = RagManager(settings, FakeEmbeddings(), top_k=3)
    ledger = LedgerStore(tmp_path / "ledger.json")
    return rag, ledger


def _run(docs: Path, tmp_path, rag: RagManager, ledger: LedgerStore) -> dict:
    clean_docs, report = run_pipeline(docs, tmp_path / "out")
    result = run_incremental_ingest(clean_docs, report, rag, ledger, docs)
    ledger.save()
    return result


def _names(paths: list[str]) -> set[str]:
    return {Path(s).name for s in paths}


class TestIncrementalIngest:
    def test_first_run_then_skip_unchanged(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text(POLICY_TEXT, encoding="utf-8")
        (docs / "b.md").write_text(TRAVEL_TEXT, encoding="utf-8")
        rag, ledger = _build_rag(tmp_path)
        scheme = rag.get("naive")

        first = _run(docs, tmp_path, rag, ledger)
        assert _names(first["rebuilt"]) == {"a.md", "b.md"}
        assert first["skipped"] == 0
        assert len(scheme) == 2
        texts_after_first = scheme.store.all_texts()

        # 第二轮内容未变：全部跳过，库中块原样保留（不重新向量化）
        second = _run(docs, tmp_path, rag, ledger)
        assert second["rebuilt"] == []
        assert second["skipped"] == 2 and second["removed"] == []
        assert scheme.store.all_texts() == texts_after_first

    def test_changed_doc_reingested_only(self, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text(POLICY_TEXT, encoding="utf-8")
        (docs / "b.md").write_text(TRAVEL_TEXT, encoding="utf-8")
        rag, ledger = _build_rag(tmp_path)
        scheme = rag.get("naive")
        _run(docs, tmp_path, rag, ledger)

        (docs / "a.md").write_text(MEETING_TEXT, encoding="utf-8")  # 仅改 A
        calls: list[str] = []
        real_add = scheme.store.add

        def spy_add(text, metadata=None):
            calls.append(text)
            real_add(text, metadata)

        monkeypatch.setattr(scheme.store, "add", spy_add)
        second = _run(docs, tmp_path, rag, ledger)

        assert _names(second["rebuilt"]) == {"a.md"}  # 只重建变更文档
        assert second["skipped"] == 1
        assert len(calls) == 1 and "会议室使用管理规定" in calls[0]  # B 未重新写入
        texts = scheme.store.all_texts()
        assert any("会议室使用管理规定" in t for t in texts)
        assert not any("项目管理制度总则" in t for t in texts)  # 旧块已删
        assert any("差旅报销管理规定" in t for t in texts)  # 未变更文档块保留

    def test_deleted_doc_removed(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text(POLICY_TEXT, encoding="utf-8")
        (docs / "b.md").write_text(TRAVEL_TEXT, encoding="utf-8")
        rag, ledger = _build_rag(tmp_path)
        scheme = rag.get("naive")
        _run(docs, tmp_path, rag, ledger)

        (docs / "b.md").unlink()
        second = _run(docs, tmp_path, rag, ledger)

        assert _names(second["removed"]) == {"b.md"}
        assert len(scheme) == 1
        assert not any("差旅报销管理规定" in t for t in scheme.store.all_texts())
        assert ledger.get(str(docs / "b.md")) is None  # 台账同步移除

    def test_downgraded_doc_removed(self, tmp_path):
        """文档本轮被判 DLQ（乱码）：旧块清理、台账移除，而非残留过期内容。"""
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text(POLICY_TEXT, encoding="utf-8")
        (docs / "b.md").write_text(TRAVEL_TEXT, encoding="utf-8")
        rag, ledger = _build_rag(tmp_path)
        scheme = rag.get("naive")
        _run(docs, tmp_path, rag, ledger)

        (docs / "b.md").write_text(("������ 乱码片段" * 20), encoding="utf-8")
        second = _run(docs, tmp_path, rag, ledger)

        assert _names(second["removed"]) == {"b.md"}
        assert len(scheme) == 1
        assert not any("差旅报销管理规定" in t for t in scheme.store.all_texts())

    def test_container_child_removed(self, tmp_path):
        """容器内子文档被移除：清理该子文档旧块，其余子文档跳过。"""
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.zip").write_bytes(
            _zip_bytes({"one.md": POLICY_TEXT.encode(), "two.md": TRAVEL_TEXT.encode()})
        )
        rag, ledger = _build_rag(tmp_path)
        scheme = rag.get("naive")
        _run(docs, tmp_path, rag, ledger)
        assert len(scheme) == 2

        (docs / "a.zip").write_bytes(_zip_bytes({"one.md": POLICY_TEXT.encode()}))
        second = _run(docs, tmp_path, rag, ledger)

        assert _names(second["removed"]) == {"two.md"}
        assert second["skipped"] == 1  # one.md 内容未变
        assert len(scheme) == 1
        assert any("项目管理制度总则" in t for t in scheme.store.all_texts())
