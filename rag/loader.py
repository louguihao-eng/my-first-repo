# =============================================================================
# rag/loader.py — 知识库加载器
# 扫描 knowledge 目录 → 读 PDF/TXT → 去重 → 存入 ChromaDB
# =============================================================================

import os
import hashlib
import pdfplumber
from langchain_core.documents import Document
from rag.store import rag_store
from core.config import settings
from loguru import logger


class KnowledgeLoader:
    def __init__(self):
        self.md5_file = os.path.join(settings.knowledge_dir, "loaded.md5")

    def _read_txt(self, path: str) -> list[Document]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            return []
        filename = os.path.basename(path)
        return [Document(page_content=text, metadata={"source": filename})]

    def _read_pdf(self, path: str) -> list[Document]:
        docs = []
        filename = os.path.basename(path)
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(
                        page_content=text.strip(),
                        metadata={"source": filename, "page": i + 1},
                    ))
        return docs

    def _file_md5(self, path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _already_loaded(self, md5: str) -> bool:
        if not os.path.exists(self.md5_file):
            return False
        with open(self.md5_file, "r") as f:
            return md5 in f.read()

    def _mark_loaded(self, md5: str):
        with open(self.md5_file, "a") as f:
            f.write(md5 + "\n")

    def run(self):
        """扫描知识库目录，加载新文件"""
        if not os.path.exists(settings.knowledge_dir):
            logger.warning(f"知识库目录不存在: {settings.knowledge_dir}")
            return

        files = [
            f for f in os.listdir(settings.knowledge_dir)
            if f.endswith((".txt", ".pdf"))
        ]
        if not files:
            logger.info("知识库目录为空")
            return

        for filename in files:
            path = os.path.join(settings.knowledge_dir, filename)
            md5 = self._file_md5(path)

            if self._already_loaded(md5):
                logger.info(f"跳过已加载: {filename}")
                continue

            try:
                if filename.endswith(".txt"):
                    docs = self._read_txt(path)
                else:
                    docs = self._read_pdf(path)

                if not docs:
                    logger.warning(f"文件无有效内容: {filename}")
                    continue

                count = rag_store.load_documents(docs)
                self._mark_loaded(md5)
                logger.info(f"加载成功: {filename} → {count} 个分片")
            except Exception as e:
                logger.error(f"加载失败 {filename}: {e}")
