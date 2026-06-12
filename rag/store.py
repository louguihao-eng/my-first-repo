# =============================================================================
# store.py —— ChromaDB 向量库封装（升级版：重写 + 混合检索 + 重排序）
# 通俗理解：这个文件就是一个"智能搜索引擎"，把知识库里的内容变成
# 向量存起来，用户提问时用最先进的方式找到最相关的几段文字。
# =============================================================================

# jieba：中文分词库，比如把"客厅用什么灯好"切成["客厅","用","什么","灯","好"]
import jieba
# BM25Okapi：经典的关键词匹配算法，擅长精确词匹配（比如搜"IP65防水"）
from rank_bm25 import BM25Okapi
# Chroma：向量数据库，把文字变成向量后存进去，支持相似度搜索
from langchain_chroma import Chroma
# Document：LangChain 里的"文档"对象，包含文字内容（page_content）和附加信息（metadata）
from langchain_core.documents import Document
# RecursiveCharacterTextSplitter：把长文章切成小块的切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# DashScopeEmbeddings：阿里云的文本转向量服务，也是"文字 → 数字串"的翻译官
from langchain_community.embeddings import DashScopeEmbeddings
# settings：整个项目的配置中心（读了 .env 里的所有配置项）
from core.config import settings
# QueryRewriter：查询重写器，把"我家客厅有点暗"翻译成"客厅吸顶灯 高瓦数"这种检索词
from rag.rewriter import QueryRewriter
# loguru：日志库，比 Python 自带的 logging 好用很多
from loguru import logger


class RagStore:
    """RAG 知识库的管理类，负责：存文档、搜文档、重排序"""

    def __init__(self):
        # === 1. 创建"文字转向量"的翻译官 ===
        # 调用阿里云 DashScope 的 API，把任意中文文本变成一个固定长度的数字数组（向量）
        self.embed_model = DashScopeEmbeddings(
            model=settings.embedding_model_name,          # 用的模型名，比如 text-embedding-v4
            dashscope_api_key=settings.dashscope_api_key or None,  # API 密钥，从 .env 读取
        )

        # === 2. 创建向量数据库 ChromaDB ===
        # ChromaDB 负责两件事：(a) 把向量存到磁盘 (b) 快速找出最相似的向量
        self.store = Chroma(
            collection_name=settings.chroma_collection_name,  # 集合名，类似数据库里的"表名"
            embedding_function=self.embed_model,              # 指定用哪个模型把文字转成向量
            persist_directory=settings.chroma_persist_dir,    # 向量数据存到磁盘的哪个文件夹
        )

        # === 3. 创建文本切分器 ===
        # 一篇2000字的文章太长了，需要切成小块（比如每块300字）才能高效检索
        # separators 是切分的优先级：先按段落切，再按句子切，最后按标点切
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,        # 每块最多多少字（默认300）
            chunk_overlap=settings.chunk_overlap,  # 相邻两块重复多少字（默认30），防止关键信息刚好被切断
            separators=["\n\n", "\n", "\u3002", "\uff1b", "\uff0c", "\u3001", " ", ""],
            # 上面这行是切分符号，优先级从左到右：
            # \n\n=空行（段落边界）  \n=换行  \u3002=。  \uff1b=；  \uff0c=，  \u3001=、  空格  无分隔符
        )

        # === 4. 懒加载占位 ===
        # 这些组件不急着初始化，等到真正要用的时候再创建（省内存、省启动时间）
        self._rewriter = None       # 查询重写器，真正用到时才创建
        self._reranker = None       # FlashRank 重排序器，真正用到时才创建（首次会下载模型）
        self._bm25_index = None     # BM25 关键词索引，存了文档后才构建
        self._bm25_docs = []        # BM25 索引对应的原始文档列表

    # =========================================================================
    # 查询重写器 —— 懒加载模式
    # @property 的作用：让你用 rag_store.rewriter 就能拿到重写器，
    # 第一次访问时自动创建，之后直接返回已创建的实例
    # =========================================================================
    @property
    def rewriter(self):
        if self._rewriter is None:              # 还没创建过？
            self._rewriter = QueryRewriter()    # 那就现在创建一个
        return self._rewriter                  # 返回重写器（可能是刚创建的，也可能是之前的）

    # =========================================================================
    # 重排序器 —— 懒加载模式
    # FlashRank 是一个轻量级 Cross-Encoder 模型（约200MB）：
    # 它能精确判断"用户问题"和"文档内容"到底有多匹配，
    # 比向量相似度准得多，但速度慢一些，所以只对候选文档做精排
    # =========================================================================
    @property
    def reranker(self):
        if self._reranker is None and settings.rag_rerank_enabled:  # 还没创建 + 功能开关是开的？
            from flashrank import Ranker                             # 导入 FlashRank
            self._reranker = Ranker()                                # 创建排序器（首次会自动下载模型）
        return self._reranker                                       # 返回排序器

    # =========================================================================
    # 文档加载 —— 把知识库文章存入向量库
    # =========================================================================
    def get_retriever(self):
        """返回一个 LangChain 标准的检索器对象，有些框架需要这个接口"""
        return self.store.as_retriever(search_kwargs={"k": settings.rag_top_k})

    def load_documents(self, documents: list[Document]):
        """
        把一批文档存入知识库，流程：
        文档 → 切小块 → 每块转向量 → 存入 ChromaDB → 重建 BM25 关键词索引
        返回：一共切了多少块
        """
        if not documents:                  # 没传文档？直接返回0
            return 0
        chunks = self.splitter.split_documents(documents)  # 把长文档切成小块
        if not chunks:                     # 切完是空的？（比如文档全空白）
            return 0
        self.store.add_documents(chunks)   # 把切好的块转成向量，存入 ChromaDB
        if settings.rag_hybrid_search_enabled:  # 如果开启了混合检索
            self._rebuild_bm25()                # 重建 BM25 关键词索引
        return len(chunks)                 # 告诉调用方一共存了多少块

    def add_documents(self, documents: list[Document]):
        """add_documents 是 load_documents 的别名，兼容旧代码的调用习惯"""
        return self.load_documents(documents)

    # =========================================================================
    # BM25 关键词检索 —— "精确匹配"这条腿
    # 稠密检索（向量）擅长"语义相近"，比如"亮点儿"能找到"高瓦数"
    # 稀疏检索（BM25）擅长"精确匹配"，比如搜索"IP65"一定能找到写"IP65"的文档
    # =========================================================================
    def _rebuild_bm25(self):
        """
        从 ChromaDB 里拿出所有已存的文档，重建 BM25 关键词索引。
        什么时候需要重建？往知识库里加了新文档之后。
        """
        try:
            result = self.store.get()           # 从 ChromaDB 取出全部文档
            if not result or not result.get("documents"):  # 库里没东西？
                return                                      # 那就直接返回
            corpus = list(result["documents"])  # 提取所有文本内容
            metadatas = result.get("metadatas", []) or []    # 提取所有元数据（可能为空）
            self._bm25_docs = [                             # 把文本+元数据重新组装成 Document
                Document(page_content=text, metadata=meta if meta else {})  # 如果元数据为空，用空字典代替
                for text, meta in zip(corpus, metadatas)    # zip：把文本和元数据一一配对
            ]
            # 对每个文档做中文分词，比如"客厅吸顶灯推荐" → ["客厅","吸顶灯","推荐"]
            tokenized = [list(jieba.cut_for_search(doc)) for     doc in corpus]
            self._bm25_index = BM25Okapi(tokenized)         # 用分词结果构建 BM25 索引
            logger.info(f"BM25 索引重建完成: {len(corpus)} 个文档")   # 日志记录
        except Exception as e:
            logger.warning(f"BM25 索引重建失败: {e}")       # 失败了也不报错，只是日志警告

    def _sparse_search(self, query: str, k: int) -> list:
        """
        BM25 关键词检索。
        输入：用户查询 + 要返回几条
        输出：[(文档, 匹配分数), ...]，按分数从高到低排列
        """
        if not self._bm25_index:                        # 索引还没构建？
            return []                                     # 返回空列表
        tokenized = list(jieba.cut_for_search(query))     # 对查询做中文分词
        scores = self._bm25_index.get_scores(tokenized)   # 计算每个文档和查询的匹配分数
        # 按分数从高到低排序，取前 k 个
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        # 返回 (文档, 分数) 的列表，分数为0的不返回（说明毫无关联）
        return [(self._bm25_docs[i], float(score)) for i, score in indexed if score > 0]

    @staticmethod
    def _rrf_fusion(dense_results, sparse_results, k=60):
        """
        RRF（倒数排名融合）—— 把稠密检索和稀疏检索的结果合并成一份。

        通俗理解：两个评委各自给文档打了排名，怎么合并？
        RRF 的做法是：排名越靠前，得分越高（用 1/(60+排名) 计算），
        然后每个文档把两次得分加起来，谁总分高谁排前面。

        参数 k=60 是经验值，防止排名波动太大。
        """
        score_map = {}   # 文档ID → 总分
        doc_map = {}     # 文档ID → 文档对象（避免重复存）

        # 第一轮：遍历稠密检索（向量相似度）结果
        for rank, doc in enumerate(dense_results):
            key = doc.page_content[:120]              # 取文档前120个字作为唯一ID
            doc_map[key] = doc                        # 记录文档对象
            score_map[key] = score_map.get(key, 0) + 1.0 / (k + rank + 1)  # 累加 RRF 分数

        # 第二轮：遍历稀疏检索（关键词匹配）结果
        for rank, (doc, _score) in enumerate(sparse_results):
            key = doc.page_content[:120]              # 同样的方式生成ID
            if key not in doc_map:                    # 稠密检索里没出现过的文档？
                doc_map[key] = doc                      # 新文档加进去
            score_map[key] = score_map.get(key, 0) + 1.0 / (k + rank + 1)  # 累加 RRF 分数

        sorted_keys = sorted(score_map, key=score_map.get, reverse=True)   # 按总分从高到低排序
        return [doc_map[key] for key in sorted_keys]                       # 按排序结果返回文档列表

    # =========================================================================
    # 检索主入口 —— 这就是整个"智能搜索引擎"的大门
    # 输入一个问题，走完整管道，返回最相关的几段知识
    # =========================================================================
    def search(self, query: str, k: int | None = None) -> list[Document]:
        """
        完整的检索管道——从收到用户问题到返回最佳文档的全过程。

        一条查询走完的流程：
        ① 查询重写：把口语变成检索关键词
        ② 双路检索：向量检索 + 关键词检索，两路同时跑
        ③ RRF 融合：把两路结果合并去重
        ④ 重排序：用精排模型对候选文档重新打分
        """
        k = k or settings.rag_top_k       # 如果没指定返回几条，用配置里的默认值（通常5条）

        # ---- Step 0: 查询重写 ----
        # 比如用户说"我想给客厅换个亮点的灯"，重写成"客厅吸顶灯 高瓦数"
        # 这样检索命中率会高很多
        if settings.rag_rewrite_enabled:
            queries = self.rewriter.rewrite(query)  # 调用 LLM 生成1~3个检索短语
        else:
            queries = [query]                       # 没开启重写？直接用原始问题搜

        # ---- Step 1: 多查询检索 + 去重 ----
        all_candidates = []   # 存放所有候选文档
        seen = set()          # 用作文档去重：同一个文档的前120字放进这个集合

        for q in queries:
            # 1a. 稠密检索（向量相似度找语义相近的文档）
            dense_docs = self.store.similarity_search(q, k=k * 2)

            # 1b. 如果开启了混合检索 + BM25 索引存在
            if settings.rag_hybrid_search_enabled and self._bm25_index:
                sparse_docs = self._sparse_search(q, k=k * 2)          # 稀疏检索（关键词匹配）
                fused = self._rrf_fusion(dense_docs, sparse_docs)      # RRF 融合两路结果
            else:
                fused = dense_docs                                      # 没开混合检索？只用向量结果

            # 1c. 去重后加入候选列表
            # 多条重写查询可能搜到同一篇文档，用 seen 集合去重
            for doc in fused:
                key = doc.page_content[:120]         # 用文档前120字做唯一标识
                if key not in seen:                   # 没见过这个文档？
                    seen.add(key)                      # 标记为"已见"
                    all_candidates.append(doc)         # 加入候选列表

        if not all_candidates:                       # 一个文档都没找到？
            return []

        # ---- Step 2: 重排序 ----
        # 候选文档可能不少（多查询导致的），用 FlashRank 精排一下
        # 条件：功能开启 + 重排序器已创建 + 候选数比需要的多
        if settings.rag_rerank_enabled and self.reranker and len(all_candidates) > k:
            try:
                from flashrank import RerankRequest                     # 导入请求类
                # 构造重排序请求：原始问题 + 所有候选文档的文本
                request = RerankRequest(
                    query=query,
                    passages=[doc.page_content for doc in all_candidates],
                )
                results = self.reranker.rerank(request)                 # 执行重排序
                # 取精排后的前 k 个文档返回
                return [all_candidates[r["index"]] for r in results[:k]]
            except Exception as e:
                logger.warning(f"重排序失败，回退到原始顺序: {e}")      # 出错了也别崩，用原来的顺序返回

        return all_candidates[:k]                    # 没走到重排序？直接返回前k个


# === 全局单例 ===
# 整个程序只需要一个 RagStore 实例，所有地方共用这一个。
# 这样做的好处：BM25 索引、重排序器都只初始化一次，不会重复创建。
rag_store = RagStore()