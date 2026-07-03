import os, sys

# リポジトリ直下を import パスに追加（共通モジュール rag_common を参照するため）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from rag_common import load_db, tokenize_ja, build_hybrid_retriever

# コマンドライン引数、なければ既定のクエリで検索
#   「私的使用のための複製」は第30条の見出しそのもの。字句一致が効く BM25 の得意技。
query = sys.argv[1] if len(sys.argv) > 1 else "私的使用のための複製"
K = 3

# --- 密ベクトルのみ（e5 + FAISS）---
dense = load_db('hourei/hourei.db').as_retriever(search_kwargs = {'k': K})

# --- BM25 のみ（保存チャンクから索引化）---
passages = []
with open('hourei/hourei_passages.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        passages.append(Document(page_content = r['text'], metadata = r['metadata']))
bm25 = BM25Retriever.from_documents(passages, preprocess_func = tokenize_ja)
bm25.k = K

# --- ハイブリッド（密 + BM25 を RRF 融合）---
hybrid = build_hybrid_retriever('hourei/hourei.db', 'hourei/hourei_passages.jsonl', k = K)


def show(label, retriever):
    print(f"\n===== {label} =====")
    for i, doc in enumerate(retriever.invoke(query), 1):
        m = doc.metadata
        # 出典は「条番号 見出し 項番号」まで示す（項ごとに分けたときは項番号が入る）。
        cite = ' '.join(x for x in [m.get('law', ''), m.get('article', ''),
                                    m.get('caption', ''), m.get('paragraph', '')] if x)
        print(f"[{i}] {cite}")
        print("    " + doc.page_content[:60].replace("\n", "") + " …")


print(f"検索クエリ: {query}")
show("密ベクトルのみ (e5+FAISS)", dense)
show("BM25 のみ (字句一致)", bm25)
show("ハイブリッド (RRF 融合)", hybrid)
