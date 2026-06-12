# =============================================================================
# product_query.py — 商品信息查询工具
# =============================================================================

from langchain_core.tools import tool
from db.database import search_products, get_product_categories
from loguru import logger


@tool
def query_products(keyword: str = "", category: str = "") -> str:
    """查询灯具商品信息，包括价格、功率、适用面积、材质、质保等。

    参数:
        keyword: 商品名称关键词，如 "吸顶灯"、"台灯"，可以为空
        category: 商品分类，如 "吸顶灯"、"吊灯"、"射灯"、"台灯"、"筒灯"、"壁灯"、"户外灯"，可以为空
    """
    try:
        results = search_products(keyword=keyword, category=category)
        if not results:
            cats = get_product_categories()
            return f"未找到匹配商品。可选的分类有: {', '.join(cats)}"

        lines = []
        for p in results[:5]:
            lines.append(
                f"【{p['name']}】¥{p['price']} | {p['wattage']} | "
                f"适用{p['room_size']} | {p['material']} | {p['warranty']}\n"
                f"  简介: {p['description']}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        logger.error(f"商品查询失败: {e}")
        return f"查询失败: {e}"
