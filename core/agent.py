# =============================================================================
# agent.py -- Agent 核心（升级版）
# 改动点：
#   1. create_agent → create_react_agent（LangGraph 新版 API）
#   2. stream_mode="messages" → 暴露 tool_call / tool_result 事件
#   3. 接入 LangFuse 可观测性（可选，未配置自动跳过）
#   4. 动态提示词：根据对话阶段自动拼接不同指令
# =============================================================================

import os
import sqlite3
# create_react_agent：LangGraph 新版 agent 工厂，比旧版 create_agent 更好控制 agent 循环
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
# 消息类型：用来区分"AI 回复文字"、"AI 要调工具"、"工具返回结果"
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from core.config import settings
from tools.rag_search import search_knowledge_base
from tools.product_query import query_products
from tools.order_query import query_orders
from prompts.builder import build_system_prompt
from loguru import logger


# ---------------------------------------------------------------------------
# 模型工厂
# ---------------------------------------------------------------------------
def _build_model():
    """创建 DeepSeek 大模型实例，每次对话都靠它来理解和生成文字"""
    return ChatDeepSeek(
        model=settings.chat_model_name,              # 模型名，比如 deepseek-v4-pro
        api_key=settings.deepseek_api_key or None,   # API 密钥（.env 里读取）
    )


# ---------------------------------------------------------------------------
# 动态提示词回调（传给 create_react_agent）
# create_react_agent 每次推理前会调用这个函数，拿到当前最新的系统指令
# ---------------------------------------------------------------------------
def _dynamic_prompt(state: dict) -> list[dict]:
    """
    根据对话历史判断当前场景，动态拼接合适的系统提示词。
    state["messages"] 是到目前为止的全部消息记录。
    """
    messages = state.get("messages", [])
    context = {}

    # 判断是否首次对话：只有系统消息 + 当前用户消息 = 算新用户
    human_count = sum(1 for m in messages
                       if getattr(m, "type", "") == "human")
    if human_count <= 1:
        context["is_new_user"] = True

    # 判断最近一次 tool 是什么（用于追加针对性规则）
    tool_names = [getattr(m, "name", "") for m in messages if isinstance(m, ToolMessage)]
    if tool_names:
        context["last_tool"] = tool_names[-1]

    # 组装最终 prompt
    return [{"role": "system", "content": build_system_prompt(context)}]


# ---------------------------------------------------------------------------
# Checkpoint 数据库路径
# ---------------------------------------------------------------------------
CHECKPOINT_DB = os.path.join(os.path.dirname(settings.lamp_db_path), "checkpoints.db")


# ---------------------------------------------------------------------------
# LampAgent —— 整个智能客服的大脑
# ---------------------------------------------------------------------------
class LampAgent:
    def __init__(self):
        # 确保 checkpoint 文件所在目录存在
        os.makedirs(os.path.dirname(CHECKPOINT_DB), exist_ok=True)
        # SQLite 连接（存对话历史的 checkpoint）
        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)

        # ---- LangFuse 可观测性（可选） ----
        # 只有同时配置了 enabled=true 和 public_key 才会启用
        # 没配置的话完全不影响正常运行
        self.langfuse_handler = None
        if settings.langfuse_enabled and settings.langfuse_public_key:
            try:
                from langfuse.callback import CallbackHandler
                self.langfuse_handler = CallbackHandler(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                logger.info("LangFuse 追踪已启用")
            except Exception as e:
                logger.warning(f"LangFuse 初始化失败，跳过: {e}")

        # ---- 创建 ReAct Agent ----
        # create_react_agent 做了三件事：
        #   1. 接到用户问题 → 决定是否要调工具
        #   2. 如果需要 → 调工具 → 拿到结果 → 再思考
        #   3. 不需要了 → 生成最终回答
        self.graph = create_react_agent(
            model=_build_model(),          # 大模型
            tools=[                        # 三个工具
                search_knowledge_base,     #   RAG 知识库搜索
                query_products,            #   商品信息查询
                query_orders,              #   订单状态查询
            ],
            checkpointer=SqliteSaver(conn),  # 对话记忆持久化
            prompt=_dynamic_prompt,        # 动态提示词（每次推理前都重新生成）
        )

    # =========================================================================
    # execute —— 流式执行，返回结构化事件
    # =========================================================================
    def execute(self, query: str, thread_id: str):
        """
        输入：用户问题 + 对话线程ID
        产出：一个接一个的事件 dict，可能包含：
          {"type": "text",        "content": "..."}       ← AI 在逐字输出
          {"type": "tool_call",   "name": "...", "args": {...}}  ← AI 决定调工具
          {"type": "tool_result", "name": "...", "content": "..."} ← 工具返回了结果
        """
        # 构造 LangGraph 运行配置
        callbacks = [self.langfuse_handler] if self.langfuse_handler else []
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks,          # LangFuse 通过这个钩子自动追踪
        }

        # stream_mode="messages" 的含义：
        # 每产生一条新消息（不管什么类型），就立刻 yield 出来
        # 对比旧的 "values" 模式：values 返回整个对话状态，"messages" 更细粒度
        for chunk in self.graph.stream(
            {"messages": [{"role": "user", "content": query}]},
            config,
            stream_mode="messages",
        ):
            # chunk 是 (message, metadata) 元组
            msg = chunk[0] if isinstance(chunk, tuple) else chunk

            # 情况1：AI 正在逐字输出文本（流式 token）
            if isinstance(msg, AIMessageChunk) and msg.content:
                yield {"type": "text", "content": msg.content}

            # 情况2：AI 输出了完整的消息（可能是 tool_calls 或者纯文本）
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # AI 决定调用工具 → 把每个工具调用作为事件发出
                    for tc in msg.tool_calls:
                        yield {
                            "type": "tool_call",
                            "name": tc["name"],            # 工具名，比如 "query_products"
                            "args": tc.get("args", {}),    # 调用参数，比如 {"keyword": "吸顶灯"}
                        }
                elif msg.content:
                    # 纯文本消息（非流式场景，作为兜底）
                    yield {"type": "text", "content": msg.content}

            # 情况3：工具执行完毕，返回了结果
            elif isinstance(msg, ToolMessage):
                yield {
                    "type": "tool_result",
                    "name": msg.name,
                    # 只截取前200字符展示，避免返回内容太长把 UI 撑爆
                    "content": str(msg.content)[:200],
                }

    # =========================================================================
    # get_messages —— 从 checkpoint 读取对话历史
    # =========================================================================
    def get_messages(self, thread_id: str) -> list:
        """用 thread_id 从持久化的 checkpoint 里把之前的对话记录拉出来"""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        if state and state.values and "messages" in state.values:
            return state.values["messages"]
        return []