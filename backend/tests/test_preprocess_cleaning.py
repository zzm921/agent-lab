"""清洗五阶段单元测试：归一化 / 页眉页脚 / 乱码 / 去重 / 质量评分。"""
from app.rag.preprocess.cleaning.boilerplate import remove_boilerplate
from app.rag.preprocess.cleaning.dedup import find_duplicates
from app.rag.preprocess.cleaning.garble import check_garble
from app.rag.preprocess.cleaning.normalizer import normalize
from app.rag.preprocess.cleaning.quality import score_text
from app.rag.preprocess.models import ParsedDocument, ParsedElement


class TestNormalize:
    def test_removes_zero_width_and_control(self):
        text = "员工\u200b手册\ufeff：考勤规定\x07。"
        result, stats = normalize(text)
        assert "\u200b" not in result and "\ufeff" not in result and "\x07" not in result
        assert stats["removed_control"] == 1

    def test_merges_broken_line(self):
        # 行尾无标点 + 次行为正文续写 → 合并，词不被硬换行切断
        result, stats = normalize("员工严重违反规章\n制度的，可以解除合同。")
        assert "规章制度" in result.replace("\n", "") or "规章制度" in result
        assert stats["merged_lines"] == 1

    def test_no_merge_after_punctuation(self):
        result, stats = normalize("第一条 员工应当遵守制度。\n第二条 考勤以系统为准。")
        assert stats["merged_lines"] == 0
        assert result.count("\n") == 1

    def test_no_merge_into_title(self):
        result, stats = normalize("# 考勤管理制度\n员工应当遵守。")
        assert stats["merged_lines"] == 0


class TestBoilerplate:
    def _doc_with_header(self) -> ParsedDocument:
        """4 页文档：每页重复页眉 + 各页正文；末页底部一次性的签名行。"""
        elements = []
        for page in range(1, 5):
            elements.append(ParsedElement(type="page_marker", text="", page=page))
            body = f"第{page}条 差旅费用凭票据实报销，须在十个工作日内提交。"
            tail = "审批人：张三 2024年1月1日" if page == 4 else f"第{page}条 附则说明内容。"
            elements.append(
                ParsedElement(
                    type="text",
                    text=f"云帆科技 内部机密 文件编号：ZD-2024-001\n{body}\n{tail}",
                    page=page,
                )
            )
        return ParsedDocument(elements=elements, page_count=4)

    def test_repeated_header_removed_unique_signature_kept(self):
        parsed = self._doc_with_header()
        elements, stats = remove_boilerplate(parsed)
        text = "\n".join(el.text for el in elements)
        assert "内部机密" not in text
        assert stats["removed_boilerplate_lines"] == 4
        assert "审批人：张三" in text  # 只出现一次的签名行不误删

    def test_page_no_removed_without_page_structure(self):
        parsed = ParsedDocument(
            elements=[ParsedElement(type="text", text="正文内容如下。\n第 3 页 / 共 4 页")]
        )
        elements, stats = remove_boilerplate(parsed)
        assert stats["removed_page_no_lines"] == 1
        assert elements[0].text == "正文内容如下。"


class TestGarble:
    def test_mojibake_detected(self):
        garbled = "æ–‡ä»¶ç®¡ç†åˆ¶åº¦è§„å®šï¼Œå'˜å·¥å¿…é¡»éµå®ˆã€‚" * 3
        is_garbled, stats = check_garble(garbled)
        assert is_garbled
        assert "mojibake" in stats["reason"]

    def test_replacement_char_detected(self):
        is_garbled, _ = check_garble("正常文字" + "�" * 10)
        assert is_garbled

    def test_empty_detected(self):
        is_garbled, _ = check_garble("   \n  ")
        assert is_garbled

    def test_good_text_passes(self):
        is_garbled, _ = check_garble("员工因公出差需提前三天在 OA 系统提交差旅申请。")
        assert not is_garbled


class TestDedup:
    def test_exact_duplicate(self):
        text = "员工手册内容：年假五天，病假需要医院证明，报销在十个工作日内提交材料。"
        # 旧→新传入：旧版（下标 0）被新版（下标 1）覆盖
        dup = find_duplicates([text, text])
        assert dup == {0: 1}

    def test_near_duplicate_detected(self):
        v1 = "考勤制度规定工作时间为九时至十八时，迟到三十分钟记旷工半日，每月三次迟到取消全勤奖，年假按工龄五到十五天，加班按倍数结算工资。" * 2
        v2 = "考勤制度规定工作时间为九时至十八时，迟到三十分钟记旷工半日，每月四次迟到取消全勤奖，年假按工龄五到十五天，加班按倍数结算工资。" * 2
        dup = find_duplicates([v1, v2])
        assert dup == {0: 1}  # v1（旧版）判定为被 v2 覆盖

    def test_distinct_documents_kept(self):
        texts = ["差旅制度：住宿一线城市每晚不超过四百元，市内交通单日上限八十元。" * 2,
                 "报销制度：材料须在返回后十个工作日内提交，逾期视为放弃报销权利。" * 2]
        assert find_duplicates(texts) == {}


class TestQuality:
    def test_good_document_high_score(self):
        good = (
            "员工因公出差需提前三天在 OA 系统提交差旅申请，注明出差地点、事由与预计费用。\n"
            "住宿标准按职级执行，普通员工一线城市每晚不超过四百元，其他城市不超过三百元。\n"
            "市内交通费凭票据实报销，单日上限八十元，出差期间原则上不安排宴请活动。\n"
            "报销材料须在返回后十个工作日内提交，包括行程单、发票与审批单，逾期视为放弃。"
        ) * 2
        score, _ = score_text(good)
        assert score >= 70

    def test_log_fragment_low_score(self):
        log = "[2024-01-15 10:23:01] INFO service started\n[2024-01-15 10:23:02] WARN timeout\n" * 5
        score, _ = score_text(log)
        assert score < 50

    def test_empty_zero(self):
        assert score_text("") == (0, {"chars": 0})
