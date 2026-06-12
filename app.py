# =============================================================================
# app.py -- 光屿灯具智能客服 前端入口（升级版）
# 启动: streamlit run app.py
#
# 本次升级：
#   1. 增量消息同步 —— 只在首次加载从 checkpoint 全量同步，之后增量追加
#   2. 工具调用可视化 —— 用户能看到 AI 正在查商品库/翻知识库
#   3. LangFuse 追踪 —— 每轮对话的完整链路自动上报（需配置 key）
# =============================================================================

import streamlit as st
from core.agent import LampAgent
from core.session import SessionManager
from db.database import seed_demo_data
from rag.loader import KnowledgeLoader
from loguru import logger


# =============================================================================
# 页面配置
# =============================================================================
st.set_page_config(page_title="光屿灯具智能客服", page_icon="💡", layout="wide")
st.title("💡 光屿灯具智能客服")
st.caption("我是小笨，你的专属灯具顾问。选购、安装、售后，随时问我。")


# =============================================================================
# 首次启动：初始化数据库演示数据 + 加载知识库
# =============================================================================
if "app_initialized" not in st.session_state:
    seed_demo_data()
    try:
        loader = KnowledgeLoader()
        loader.run()
    except Exception as e:
        logger.warning(f"知识库加载跳过: {e}")
    st.session_state["app_initialized"] = True


# =============================================================================
# 初始化 Agent 和会话管理器（存在 st.session_state 里，整个 app 生命周期只创建一次）
# =============================================================================
if "agent" not in st.session_state:
    st.session_state["agent"] = LampAgent()

if "session_mgr" not in st.session_state:
    st.session_state["session_mgr"] = SessionManager()

agent = st.session_state["agent"]
mgr   = st.session_state["session_mgr"]


# =============================================================================
# 确定当前 thread_id：优先恢复上次会话
# =============================================================================
if "thread_id" not in st.session_state:
    restored = mgr.resume_latest()
    if restored:
        st.session_state["thread_id"] = restored
    else:
        st.session_state["thread_id"] = mgr.new_session("新对话")

thread_id = st.session_state["thread_id"]


# =============================================================================
# 消息同步（★ 升级点1：增量同步）
# 旧逻辑：每次 rerun 都从 checkpoint 全量拉 + 全量覆盖 → 对话越长越慢
# 新逻辑：首次进入才全量同步，之后消息由用户输入和 AI 输出直接追加
# =============================================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not st.session_state.get("_messages_synced"):
    raw_messages = agent.get_messages(thread_id)
    synced = []
    for msg in raw_messages:
        if msg.type == "human":
            synced.append({"role": "user", "content": msg.content})
        elif msg.type == "ai" and msg.content:
            synced.append({"role": "assistant", "content": str(msg.content)})
    st.session_state["messages"] = synced
    st.session_state["_messages_synced"] = True

# 每个对话线程切换时，重置同步标记
current_session_key = f"_synced_{thread_id}"
if current_session_key not in st.session_state:
    st.session_state["messages"] = []
    raw_messages = agent.get_messages(thread_id)
    synced = []
    for msg in raw_messages:
        if msg.type == "human":
            synced.append({"role": "user", "content": msg.content})
        elif msg.type == "ai" and msg.content:
            synced.append({"role": "assistant", "content": str(msg.content)})
    st.session_state["messages"] = synced
    st.session_state[current_session_key] = True


# =============================================================================
# 侧边栏 —— 会话管理
# =============================================================================
with st.sidebar:
    st.subheader("📱 历史会话")

    if st.button("＋ 新建会话", use_container_width=True):
        new_id = mgr.new_session("新对话")
        st.session_state["thread_id"] = new_id
        st.session_state["messages"] = []
        st.session_state[f"_synced_{new_id}"] = True
        st.rerun()

    st.divider()

    sessions = mgr.list_all()
    for s in sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            label = s["title"] or "新对话"
            if s["thread_id"] == thread_id:
                label = f"🔵 {label}"
            if st.button(label, key=f"ses_{s['thread_id']}", use_container_width=True):
                st.session_state["thread_id"] = s["thread_id"]
                mgr.switch_session(s["thread_id"])
                st.session_state["messages"] = []
                st.session_state[f"_synced_{s['thread_id']}"] = True
                st.rerun()
        with col2:
            if st.button("🗏", key=f"del_{s['thread_id']}"):
                mgr.delete(s["thread_id"])
                if s["thread_id"] == thread_id:
                    restored = mgr.resume_latest()
                    if restored:
                        st.session_state["thread_id"] = restored
                        st.session_state[f"_synced_{restored}"] = True
                    else:
                        new_id = mgr.new_session("新对话")
                        st.session_state["thread_id"] = new_id
                        st.session_state[f"_synced_{new_id}"] = True
                    st.session_state["messages"] = []
                st.rerun()


# =============================================================================
# 主区域 —— 聊天记录渲染
# =============================================================================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# =============================================================================
# 输入框
# =============================================================================
prompt = st.chat_input("请输入你的问题...")

if prompt:
    # 第一条消息自动作为会话标题（取前20个字）
    if len(st.session_state["messages"]) == 0:
        title = prompt[:20] + ("..." if len(prompt) > 20 else "")
        mgr.update_title(thread_id, title)

    # 先显示用户消息
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # ------- AI 回复区域 -------
    with st.chat_message("assistant"):
        # 工具调用状态显示区（小字，在回复文字上方）
        tool_status = st.empty()
        # AI 文字流式输出区
        text_placeholder = st.empty()

        full_response = ""
        tool_calls_log = []   # 收集本轮所有工具调用记录

        # ---- 流式接收 agent 的结构化事件 ----
        for event in agent.execute(prompt, thread_id):
            if event["type"] == "text":
                # ★ 文本内容：逐字追加到输出区
                full_response += event["content"]
                text_placeholder.markdown(full_response + "▌")

            elif event["type"] == "tool_call":
                # ★ AI 决定调工具了 → 在状态区显示"正在查询..."
                tool_calls_log.append(event)
                names = [t["name"] for t in tool_calls_log if t["type"] == "tool_call"]
                if names:
                    tool_status.caption("🔍 " + " · ".join(f"正在调用 {n}" for n in names[-2:]))

            elif event["type"] == "tool_result":
                # ★ 工具执行完了 → 更新状态
                tool_calls_log.append(event)
                completed = [t for t in tool_calls_log if t["type"] == "tool_result"]
                if completed:
                    latest = completed[-1]
                    tool_status.caption(f"✅ {latest['name']} 完成")

        # ---- 流式结束，渲染最终回复 ----
        text_placeholder.markdown(full_response)

        # ----（★ 升级点2）如果有工具调用，显示可展开的详情 ----
        if tool_calls_log:
            with st.expander("🔍 查看检索过程", expanded=False):
                for te in tool_calls_log:
                    if te["type"] == "tool_call":
                        st.caption(f"📞 调用 `{te['name']}`")
                        if te.get("args"):
                            st.code(str(te["args"]), language="json")
                    elif te["type"] == "tool_result":
                        st.caption(f"📥 `{te['name']}` 返回")
                        st.text(te.get("content", "")[:300])

    # ---- 保存到消息历史 ----
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    mgr.touch()
    st.rerun()