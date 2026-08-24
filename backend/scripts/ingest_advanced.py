"""建库脚本：advanced（高级 RAG）方案——语义分块 + 混合检索。

在线服务启动前运行：将内嵌语料写入 advanced 方案的独立向量库（Qdrant/ES）。
在 backend/ 目录下执行：python scripts/ingest_advanced.py [--force]
幂等：语料未变则跳过；--force 强制清空重建。
"""
import argparse

from app.rag.ingest import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 advanced RAG 方案的向量库")
    parser.add_argument("--force", action="store_true", help="强制清空并重建（忽略语料指纹）")
    args = parser.parse_args()
    status = build_corpus(["advanced"], force=args.force)
    for entry in status:
        print(f"[advanced] {entry['collection']}: {entry['count']} 条")


if __name__ == "__main__":
    main()
