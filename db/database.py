# =============================================================================
# database.py — 业务数据库层
# 管理会话记录、商品信息、订单数据
# 首次运行时自动建表
# =============================================================================

import sqlite3
import os
from datetime import datetime
from core.config import settings


DB_PATH = settings.lamp_db_path


def _ensure_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_conn():
    """获取数据库连接（自动建表）"""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # 让查询结果可以用 dict 方式访问
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id   TEXT PRIMARY KEY,
            title       TEXT DEFAULT '新对话',
            created_at  TEXT NOT NULL,
            last_active TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS products (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            category    TEXT NOT NULL,
            price       REAL NOT NULL,
            wattage     TEXT,
            room_size   TEXT,
            material    TEXT,
            warranty    TEXT,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS orders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            product_name TEXT NOT NULL,
            quantity    INTEGER DEFAULT 1,
            status      TEXT DEFAULT '待发货',
            created_at  TEXT NOT NULL
        );
    """)
    conn.commit()


# =============================================================================
# 会话操作
# =============================================================================

def create_session(thread_id: str, title: str = "新对话"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO sessions (thread_id, title, created_at, last_active) VALUES (?, ?, ?, ?)",
        (thread_id, title, now, now),
    )
    conn.commit()


def update_session_active(thread_id: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    conn.execute("UPDATE sessions SET last_active = ? WHERE thread_id = ?", (now, thread_id))
    conn.commit()


def update_session_title(thread_id: str, title: str):
    conn = get_conn()
    conn.execute("UPDATE sessions SET title = ? WHERE thread_id = ?", (title, thread_id))
    conn.commit()


def list_sessions(limit: int = 20) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY last_active DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_latest_session() -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM sessions ORDER BY last_active DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def delete_session(thread_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
    conn.commit()


# =============================================================================
# 商品查询
# =============================================================================

def search_products(keyword: str = "", category: str = "", limit: int = 10) -> list[dict]:
    conn = get_conn()
    query = "SELECT * FROM products WHERE 1=1"
    params: list = []
    if keyword:
        query += " AND (name LIKE ? OR description LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    if category:
        query += " AND category = ?"
        params.append(category)
    query += " LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_product_by_id(product_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    return dict(row) if row else None


def get_product_categories() -> list[str]:
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT category FROM products ORDER BY category").fetchall()
    return [r["category"] for r in rows]


# =============================================================================
# 订单查询
# =============================================================================

def search_orders(user_id: str = "", limit: int = 20) -> list[dict]:
    conn = get_conn()
    if user_id:
        rows = conn.execute(
            "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
# 初始化演示数据
# =============================================================================

def seed_demo_data():
    """插入演示商品和订单，方便测试"""
    conn = get_conn()

    existing = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if existing > 0:
        return

    products = [
        ("星辰吸顶灯-经典款", "吸顶灯", 299.00, "36W", "15-25㎡", "铁艺+亚克力", "3年质保", "简约现代设计，三档色温调节，适合卧室、书房"),
        ("星辰吸顶灯-Pro", "吸顶灯", 499.00, "60W", "25-40㎡", "铝合金+亚克力", "5年质保", "无极调光调色，支持天猫精灵/小爱同学控制，适合客厅"),
        ("北欧分子吊灯-7头", "吊灯", 899.00, "7×5W", "15-30㎡", "黄铜+玻璃", "2年质保", "北欧轻奢风格，手工玻璃灯罩，适合餐厅、吧台"),
        ("极简轨道射灯-3米套装", "射灯", 399.00, "4×12W", "10-20㎡", "航空铝", "3年质保", "COB光源，RA>95高显指，适合画廊、展厅、服装店"),
        ("智能护眼台灯", "台灯", 199.00, "12W", "桌面", "ABS+铝合金", "1年质保", "国AA级照度，无蓝光危害，自动调光，适合学生读写"),
        ("LED筒灯嵌入式-10只装", "筒灯", 159.00, "10×7W", "多区域", "PC阻燃", "2年质保", "开孔7-8cm，4000K中性光，适合走廊、过道"),
        ("新中式壁灯", "壁灯", 259.00, "8W", "5-8㎡", "实木+布艺", "2年质保", "暖黄光氛围灯，中式禅意设计，适合床头、玄关"),
        ("户外防水壁灯", "户外灯", 349.00, "15W", "庭院", "压铸铝+钢化玻璃", "5年质保", "IP65防水防尘，光感应自动亮灭，适合庭院、门廊"),
    ]
    conn.executemany(
        "INSERT INTO products (name, category, price, wattage, room_size, material, warranty, description) VALUES (?,?,?,?,?,?,?,?)",
        products,
    )

    orders = [
        ("U1001", "星辰吸顶灯-经典款", 2, "已签收", "2026-05-20 10:00:00"),
        ("U1001", "智能护眼台灯", 1, "运输中", "2026-06-01 14:30:00"),
        ("U1002", "北欧分子吊灯-7头", 1, "待发货", "2026-06-03 09:00:00"),
        ("U1003", "LED筒灯嵌入式-10只装", 1, "已签收", "2026-05-15 16:00:00"),
        ("U1003", "极简轨道射灯-3米套装", 1, "已签收", "2026-05-15 16:05:00"),
    ]
    conn.executemany(
        "INSERT INTO orders (user_id, product_name, quantity, status, created_at) VALUES (?,?,?,?,?)",
        orders,
    )
    conn.commit()
