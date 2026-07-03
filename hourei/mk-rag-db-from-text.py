# リポジトリ直下を import パスに追加（共通モジュール rag_common を参照するため）
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from rag_common import get_embeddings

# Step1. 条文(JSONL)の読み込み。1行 = 1項（短い条は1条）。
docs = []
with open('hourei/hourei.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        docs.append(Document(page_content = r['text'], metadata = r['metadata']))

# Step2. パッセージへの分解。
#   取得段階で条・項の単位に整えてあるので、通常はそのまま1チャンクにする。
#   まれに残る極端に長い項だけ、句点区切りで大きめに再分割する保険をかける。
#   条・項のメタデータ(law/article/caption/paragraph/source)は各チャンクに引き継がれる。
text_splitter = RecursiveCharacterTextSplitter(
    separators = ['\n', '。', '、', ''],  # 文の途中で割らないよう句読点を優先する
    keep_separator = True,               # 区切りの句点を本文に残す
    chunk_size = 1200,    # チャンクの文字数（大きめにして項をなるべく丸ごと残す）
    chunk_overlap = 100,  # チャンクオーバーラップの文字数
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

print(f'{len(docs)} レコードを {len(passages)} チャンクに分割し、'
      f'hourei/hourei.db（密）と hourei/hourei_passages.jsonl（BM25用）に保存しました。')
