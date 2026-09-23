"""RAG 知识库子包（骨架，实现计划 P1）：LlamaIndex 落地，工具签名不变替换 Mock。

对外暴露三个工具工厂，agents 层换一下 import 来源即完成"换芯不换线"：
- get_product_retriever_tool  → 商品检索（原 tools/product_tools.py 的 search_product，文件已删）
- get_policy_retriever_tool   → 售后政策检索（原 aftersale_tools 的静态 REFUND_POLICY）
- get_faq_retriever_tool      → 常见问题问答对检索（search_faq）

目录约定：
- knowledge/  语料源文件（一个主题一个 md）
- storage/    索引持久化目录（运行时生成，已入 .gitignore）

依赖：uv sync --extra rag（pyproject.toml 的 optional-dependencies.rag）。
没装之前 import 本包会报 ModuleNotFoundError，属预期。

可选加分（计划 P1+）：用 LightRAG 实现同签名工具，与 LlamaIndex 版对比检索质量。
"""

from customer_service.rag.retriever import (
    get_faq_retriever_tool,
    get_policy_retriever_tool,
    get_product_retriever_tool,
)

__all__ = [
    "get_product_retriever_tool",
    "get_policy_retriever_tool",
    "get_faq_retriever_tool",
]
