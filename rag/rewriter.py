# =============================================================================
# rewriter.py -- 查询重写模块
# 将用户自然语言问题转为检索优化的关键词查询
# =============================================================================

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from core.config import settings
from loguru import logger

REWRITE_SYSTEM_PROMPT = """你是查询优化器。将用户的灯具相关问题改写为 1~3 个检索短语，每个一行。
规则：
- 提取核心实体：产品类型、房间、功率、价格区间、风格
- 补充同义词和常见说法（如"亮点儿"="高瓦数""高亮度"）
- 只输出关键词短语，不要解释、不要序号

示例输入：我家客厅大概20平，想换个亮点儿的灯
示例输出：
客厅吸顶灯 高瓦数
20平米 客厅 主灯 60W以上
客厅 LED吸顶灯 高亮度"""


class QueryRewriter:
    """用 LLM 将用户口语改写为检索友好查询"""

    def __init__(self):
        self.llm = ChatDeepSeek(
            model=settings.chat_model_name,
            api_key=settings.deepseek_api_key,
            temperature=0,
            max_tokens=150,
        )

    def rewrite(self, user_query: str) -> list[str]:
        """返回 1~3 条重写后的查询短语，失败时回退为原始查询"""
        try:
            response = self.llm.invoke([
                SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ])
            queries = [
                q.strip()
                for q in str(response.content).split("\n")
                if q.strip()
            ]
            return queries if queries else [user_query]
        except Exception as e:
            logger.warning(f"查询重写失败，使用原始查询: {e}")
            return [user_query]
