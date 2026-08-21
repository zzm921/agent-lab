"""工具与向量检索测试。"""
from datetime import datetime

import pytest

from app.memory.vector_store import VectorStore
from app.tools.calculator import calculator
from app.tools.time_now import time_now


def test_calculator_basic():
    assert calculator.invoke({"expression": "1+1"}) == "2"


def test_calculator_chinese_operators():
    assert calculator.invoke({"expression": "(137×0.85−20)÷3"}) == "32.15"


def test_calculator_unsafe_rejected():
    with pytest.raises(ValueError):
        calculator.invoke({"expression": "__import__('os')"})


def test_time_now_format():
    result = time_now.invoke({})
    datetime.strptime(result, "%Y-%m-%d %H:%M:%S")


def test_vector_store_search_topk(embeddings):
    vs = VectorStore(embeddings)
    vs.add("关于苹果的水果信息")
    vs.add("关于汽车的技术信息")
    hits = vs.search("水果", top_k=1)
    assert len(hits) == 1
    assert hits[0]["score"] >= 0.0


def test_vector_store_empty(embeddings):
    vs = VectorStore(embeddings)
    assert vs.search("任意") == []
