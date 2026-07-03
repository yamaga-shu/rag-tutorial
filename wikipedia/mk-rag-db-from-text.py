# リポジトリ直下を import パスに追加（共通モジュール rag_common を参照するため）
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from rag_common import get_embeddings

# Step1. 記事(JSONL)の読み込み
docs = []
with open('wikipedia/wikipedia.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        docs.append(Document(
            page_content=record['text'],
            metadata={'title': record['title'], 'source': record['source']},
        ))

# Step2. パッセージへの分解（Wikipedia は説明文なので joseito より大きめ）
#   メタデータ(title/source)は各チャンクに引き継がれる
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,   #チャンクの文字数
    chunk_overlap = 50, #チャンクオーバーラップの文字数
)
passages = text_splitter.split_documents(docs)

# Step3. パッセージのベクトル化
embeddings = get_embeddings()

# Step4. DBの作成（from_documents でメタデータごと保存）
db = FAISS.from_documents(passages, embeddings)
db.save_local('wikipedia/wikipedia.db')

print(f"{len(docs)} 記事を {len(passages)} チャンクに分割し、wikipedia/wikipedia.db に保存しました。")
