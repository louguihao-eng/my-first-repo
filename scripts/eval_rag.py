# =============================================================================
# eval_rag.py -- RAG 管道评测脚本
# 用法: python scripts/eval_rag.py
# 作用: 用一组测试问题来定量衡量 RAG 检索的准确率和召回率
#
# 通俗理解：
#   - 上下文精确率（context_precision）：搜出来的文档里，有几条是真正有用的？
#   - 上下文召回率（context_recall）  ：应该搜到的文档里，实际搜到了几条？
#   - 忠实度（faithfulness）         ：AI 的回答是不是基于搜到的文档，有没有瞎编？
#
# 首次运行需要 ragas 库: pip install ragas
# =============================================================================

import sys
import os

# 把项目根目录加入 sys.path，这样能 import core、rag 等模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.store import rag_store
from collections import defaultdict


# =============================================================================
# 测试数据集
# 每条包含：问题（question）、期望检索到的关键词（expected_keywords）
# 这些数据来源于 灯具选购指南.txt 的「常见问题」部分
# =============================================================================
TEST_CASES = [
    {
        "question": "灯买回去不亮怎么办？",
        "expected_keywords": ["接线", "驱动器", "不亮", "退货"],
    },
    {
        "question": "色温怎么选？",
        "expected_keywords": ["色温", "3000K", "4000K", "5000K", "卧室", "客厅"],
    },
    {
        "question": "LED灯能用多久？",
        "expected_keywords": ["LED", "寿命", "25000", "50000", "小时"],
    },
    {
        "question": "客厅20平左右选什么灯？",
        "expected_keywords": ["客厅", "20", "60W", "100W", "吸顶灯", "4000K"],
    },
    {
        "question": "卧室应该装什么灯？",
        "expected_keywords": ["卧室", "10", "20", "24W", "36W", "3000K"],
    },
    {
        "question": "餐厅吊灯装多高合适？",
        "expected_keywords": ["餐厅", "吊灯", "60", "80cm"],
    },
    {
        "question": "厨房用什么灯好？",
        "expected_keywords": ["厨房", "4000K", "5000K", "冷白光"],
    },
    {
        "question": "卫生间要买什么样的灯？",
        "expected_keywords": ["卫生间", "防水", "IP44"],
    },
    {
        "question": "买灯7天内可以退吗？",
        "expected_keywords": ["7天", "无理由", "退换", "售后"],
    },
    {
        "question": "灯具质保多久？",
        "expected_keywords": ["质保", "1", "5", "年", "易耗品", "6个月"],
    },
    {
        "question": "安装灯需要注意什么？",
        "expected_keywords": ["安装", "断电", "膨胀螺丝", "接线", "驱动器", "散热"],
    },
    {
        "question": "现代简约风格配什么灯？",
        "expected_keywords": ["现代简约", "几何线条", "吸顶灯", "轨道射灯"],
    },
    {
        "question": "北欧风装修配什么灯具？",
        "expected_keywords": ["北欧", "原木", "分子吊灯", "布艺壁灯"],
    },
    {
        "question": "新中式风格配什么灯？",
        "expected_keywords": ["新中式", "实木", "黄铜", "壁灯"],
    },
    {
        "question": "怎么判断灯够不够亮？",
        "expected_keywords": ["功率", "60W", "LED", "面积", "瓦数"],
    },
]


# =============================================================================
# 评测逻辑
# =============================================================================
def evaluate_retrieval(k: int = 5) -> dict:
    """
    对每条测试问题执行 RAG 检索，用关键词匹配来模拟"相关/不相关"判断。
    返回精确率、召回率、F1 等汇总指标。
    """
    total_precision = 0.0
    total_recall    = 0.0
    details = []

    for tc in TEST_CASES:
        question = tc["question"]
        expected = set(tc["expected_keywords"])

        # ---- 执行检索 ----
        docs = rag_store.search(question, k=k)

        # ---- 统计命中 ----
        retrieved_text = " ".join(d.page_content for d in docs)
        hit_count = sum(1 for kw in expected if kw.lower() in retrieved_text.lower())

        precision = hit_count / k if k > 0 else 0          # 搜回来的里面多少相关
        recall    = hit_count / len(expected) if expected else 0  # 应该搜到的搜到了多少

        total_precision += precision
        total_recall    += recall

        details.append({
            "question":       question,
            "precision":      round(precision, 3),
            "recall":         round(recall, 3),
            "hits":           hit_count,
            "expected_count": len(expected),
            "retrieved_count": len(docs),
        })

    n = len(TEST_CASES)
    avg_precision = total_precision / n if n > 0 else 0
    avg_recall    = total_recall / n if n > 0 else 0
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return {
        "avg_precision": round(avg_precision, 3),
        "avg_recall":    round(avg_recall, 3),
        "f1_score":      round(f1, 3),
        "total_cases":   n,
        "retrieval_k":   k,
        "details":       details,
    }


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG 检索质量评测")
    print("=" * 60)
    print()

    for k_value in [3, 5, 10]:
        result = evaluate_retrieval(k=k_value)
        print(f"--- Top K = {k_value} ---")
        print(f"  平均精确率 (Precision): {result['avg_precision']:.2%}")
        print(f"  平均召回率 (Recall):    {result['avg_recall']:.2%}")
        print(f"  F1 分数:               {result['f1_score']:.2%}")
        print()

    # 打印每条详情
    print("--- 逐条详情 (Top K=5) ---")
    result5 = evaluate_retrieval(k=5)
    for d in result5["details"]:
        bar = "█" * int(d["precision"] * 20) + "░" * (20 - int(d["precision"] * 20))
        print(f"  [{bar}] {d['question']}")
        print(f"        命中 {d['hits']}/{d['expected_count']} 关键词")

    print()
    print("评测完成。分数越高 = RAG 检索越准。")
    print("提示：修改 chunk_size、rerank_enabled 等参数后重新跑本脚本，对比分数变化。")