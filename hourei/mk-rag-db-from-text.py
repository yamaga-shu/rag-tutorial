# リポジトリ直下を import パスに追加（共通モジュール rag_common を参照するため）
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from rag_common import get_embeddings

# Step1. 条文(JSONL)の読み込み。1行 = 1条。
docs = []
with open('hourei/hourei.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        docs.append(Document(page_content = r['text'], metadata = r['metadata']))

# Step2. パッセージへの分解。
#   法令は「条」が自然な単位だが、定義規定など長い条は 500 字程度に再分割する。
#   条のメタデータ(law/article/caption/source)は各チャンクに引き継がれる。
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,    # チャンクの文字数
    chunk_overlap = 50,  # チャンクオーバーラップの文字数
)
passages = text_splitter.split_documents(docs)

# Step3. パッセージのベクトル化（密ベクトル側）
embeddings = get_embeddings()

# Step4. 密ベクトル DB（FAISS）の作成と保存
db = FAISS.from_documents(passages, embeddings)
db.save_local('hourei/hourei.db')

# Step5. BM25（疎）側は検索時にチャンクから索引を作り直すので、
#   分割済みチャンクをそのまま JSONL に保存しておく（密と疎で同じ母集団にするため）。
with open('hourei/hourei_passages.jsonl', 'w', encoding='utf-8') as f:
    for p in passages:
        f.write(json.dumps({'text': p.page_content, 'metadata': p.metadata}, ensure_ascii=False) + '\n')

print(f'{len(docs)} 条を {len(passages)} チャンクに分割し、'
      f'hourei/hourei.db（密）と hourei/hourei_passages.jsonl（BM25用）に保存しました。')
