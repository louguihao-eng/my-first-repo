# =============================================================================
# session.py — 多会话管理
# 管理多个对话会话：新建、切换、删除、恢复
# =============================================================================

import uuid
import os
from datetime import datetime
from db.database import (
    create_session, update_session_active, update_session_title,
    list_sessions, get_latest_session, delete_session,
)


class SessionManager:
    def __init__(self):
        self._current_thread_id: str | None = None
        self._current_title: str = "新对话"

    @property
    def current_thread_id(self) -> str | None:
        return self._current_thread_id

    @property
    def current_title(self) -> str:
        return self._current_title

    def new_session(self, title: str = "新对话") -> str:
        """创建新会话，返回 thread_id"""
        thread_id = str(uuid.uuid4())
        create_session(thread_id, title)
        self._current_thread_id = thread_id
        self._current_title = title
        return thread_id

    def switch_session(self, thread_id: str) -> bool:
        """切换到已有会话"""
        sessions = list_sessions()
        for s in sessions:
            if s["thread_id"] == thread_id:
                self._current_thread_id = thread_id
                self._current_title = s["title"]
                update_session_active(thread_id)
                return True
        return False

    def resume_latest(self) -> str | None:
        """恢复最近一次会话，没有则返回 None"""
        latest = get_latest_session()
        if latest:
            self._current_thread_id = latest["thread_id"]
            self._current_title = latest["title"]
            return latest["thread_id"]
        return None

    def update_title(self, thread_id: str, title: str):
        update_session_title(thread_id, title)
        if thread_id == self._current_thread_id:
            self._current_title = title

    def list_all(self) -> list[dict]:
        return list_sessions()

    def delete(self, thread_id: str):
        delete_session(thread_id)
        if thread_id == self._current_thread_id:
            self._current_thread_id = None

    def touch(self):
        if self._current_thread_id:
            update_session_active(self._current_thread_id)
