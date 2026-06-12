# =============================================================================
# rag_search.py — RAG 知识库检索工具
# =============================================================================

from langchain_core.tools import tool
from rag.store import rag_store
from loguru import logger


@tool
def search_knowledge_base(query: str) -> str:
    """搜索灯具知识库（文本资料），获取灯具选购指南、使用安装方法、售后政策等内容。

    适用场景：
        - 用户问"怎么选"、"怎么装"、"售后政策"、"注意事项"等知识性问题
        - 需要从说明文档中查找答案

    不适用场景：
        - 查具体商品价格/参数/库存 → 请用 query_products
        - 查订单状态 → 请用 query_orders

    参数:
        query: 用户的具体问题，建议包含关键词，例如 "吸顶灯适合多少平米"
    """
    try:
        docs = rag_store.search(query)
        if not docs:
            return "知识库中暂未找到相关信息。"
        parts = []
        for i, doc in enumerate(docs[:5], 1):
            src = doc.metadata.get("source", "未知来源")
            parts.append(f"[{i}] {doc.page_content}\n  ——来源: {src}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
        return f"检索失败: {e}"
