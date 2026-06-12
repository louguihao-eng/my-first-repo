# =============================================================================
# builder.py -- 提示词构建器
# 通俗理解：根据对话的上下文（新用户？查订单？），动态拼装不同的系统指令
# =============================================================================

import os


def build_system_prompt(context: dict | None = None) -> str:
    """
    读取基础提示词模板，根据对话上下文动态拼接附加指令。

    输入 context 可选字段：
      - is_new_user:   bool   是否首次对话（加自我介绍引导）
      - last_tool:     str    上一次调用的工具名（针对性补充规则）
    """
    # 1. 加载基础提示词（店铺信息、职责、工具优先级等）
    base = _read_base_prompt()

    # 2. 根据上下文拼接动态指令
    context = context or {}
    dynamic = ""

    if context.get("is_new_user"):
        dynamic += (
            "\n\n【当前场景：首次对话】"
            "\n用户是第一次来访，请："
            "\n1. 用热情友好的语气简单介绍光屿灯具（主营品类、售后政策亮点）"
            "\n2. 询问用户需要什么帮助"
        )

    if context.get("last_tool") == "query_orders":
        dynamic += (
            "\n\n【当前场景：订单查询】"
            "\n注意："
            "\n1. 如果用户还没有提供用户ID，请先礼貌地询问"
            "\n2. 不要主动泄露或暗示其他用户的订单信息"
        )

    if context.get("last_tool") == "search_knowledge_base":
        dynamic += (
            "\n\n【当前场景：知识咨询】"
            "\n注意：回答中引用知识库内容时，用通俗的语言转述，而不是直接复制原文。"
        )

    return base + dynamic


def _read_base_prompt() -> str:
    """读取 system.txt 基础模板"""
    path = os.path.join(os.path.dirname(__file__), "system.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "你是一个专业的灯具电商客服助手。"