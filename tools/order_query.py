# =============================================================================
# order_query.py — 订单查询工具
# =============================================================================

from langchain_core.tools import tool
from db.database import search_orders
from loguru import logger


@tool
def query_orders(user_id: str = "") -> str:
    """查询用户的订单信息，包括订单状态、商品名称、数量、下单时间等。

    参数:
        user_id: 用户ID，默认查询用户U1001的订单
    """
    try:
        uid = user_id.strip() if user_id else "U1001"
        results = search_orders(user_id=uid)

        if not results:
            return f"用户 {uid} 暂无订单记录。"

        lines = [f"用户 {uid} 的订单："]
        for o in results[:10]:
            lines.append(
                f"  · {o['product_name']} ×{o['quantity']} | "
                f"状态: {o['status']} | 下单时间: {o['created_at']}"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"订单查询失败: {e}")
        return f"查询失败: {e}"
