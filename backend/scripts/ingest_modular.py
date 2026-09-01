"""建库脚本：modular / agentic 方案——前置语义分类 + 语义分块。

在线服务启动前运行：将内嵌语料写入两方案的独立向量库（Qdrant/ES，同语料不同集合）。
在 backend/ 目录下执行：python scripts/ingest_modular.py [--scheme modular agentic] [--force]
缺省同时构建 modular 与 agentic；--scheme 可指定只建其中某个（如 agentic 单独补库）。
幂等：语料未变则跳过；--force 强制清空重建。
"""
import argparse

from app.rag.ingest import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 modular/agentic RAG 方案的向量库")
    parser.add_argument(
        "--scheme",
        nargs="*",
        choices=["modular", "agentic"],
        default=["modular", "agentic"],
        help="要建库的方案（缺省同时构建 modular 与 agentic）",
    )
    parser.add_argument("--force", action="store_true", help="强制清空并重建（忽略语料指纹）")
    args = parser.parse_args()
    status = build_corpus(args.scheme, force=args.force)
    for entry in status:
        print(f"[{entry['id']}] {entry['collection']}: {entry['count']} 条")


if __name__ == "__main__":
    main()