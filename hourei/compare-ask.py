import os, sys, datetime

# リポジトリ直下を import パスに追加（共通モジュール rag_common を参照するため）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from rag_common import build_dense_retriever, build_hybrid_retriever, get_llm

# ask-rag.py と同じ設定。retriever だけを差し替えて回答を比べる。
K = 4
DB = 'hourei/hourei.db'
PASSAGES = 'hourei/hourei_passages.jsonl'
OUT_PATH = 'hourei/compare/ask-compare.txt'

# 既定の比較クエリ。CLI 引数を渡せばそのクエリで比較する。
#   図書館の複製は第三十一条の各項（一部分の複製、保存、絶版等資料、公衆送信）に
#   分かれており、BM25 が同一条の各項を字句一致で拾うため、密のみとの差が出やすい。
DEFAULT_QUERIES = [
    '図書館では著作物をどこまで複製できますか？',
]

# LLM と、検索した文脈を渡すためのプロンプト（ask-rag.py と同一）
llm = get_llm()
prompt = ChatPromptTemplate.from_messages([
    ('system',
     'あなたは提示された条文だけを根拠に日本語で答える法令アシスタントです。'
     '答える際は根拠にした条番号を必ず示してください。'
     '条文に答えが無い場合は「分かりません」と答えてください。\n\n'
     '条文:\n{context}'),
    ('human', '{input}'),
])
document_chain = create_stuff_documents_chain(llm, prompt)


def build_chain(retriever):
    return create_retrieval_chain(retriever, document_chain)


# 「密のみ（ハイブリッドでない）」と「ハイブリッド」の2つのチェーンを用意
dense_chain = build_chain(build_dense_retriever(DB, k = K))
hybrid_chain = build_chain(build_hybrid_retriever(DB, PASSAGES, k = K))


def cite(doc):
    """参照した条文の出典を「法令名 条番号 見出し 項番号」で表す。"""
    m = doc.metadata
    return ' '.join(x for x in [m.get('law', ''), m.get('article', ''),
                                m.get('caption', ''), m.get('paragraph', '')] if x)


def render(label, result):
    """1つのチェーンの結果（回答＋参照条文）を文字列に整形する。"""
    lines = [f"----- {label} -----", "【回答】", result['answer'].strip(), "", "【参照した条文】"]
    for i, doc in enumerate(result['context'], 1):
        lines.append(f"[{i}] {cite(doc)}")
    return "\n".join(lines)


def compare(query):
    dense = dense_chain.invoke({'input': query})
    hybrid = hybrid_chain.invoke({'input': query})
    block = [
        "#" * 60,
        f"# 質問: {query}",
        "#" * 60,
        "",
        render("密ベクトルのみ（ハイブリッドでない）", dense),
        "",
        render("ハイブリッド（密 + BM25 を RRF 融合）", hybrid),
    ]
    return "\n".join(block)


queries = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_QUERIES

header = f"著作権法 RAG：密のみ vs ハイブリッド 回答比較（k={K}）\n生成: {datetime.date.today()}\n"
blocks = [header]
for q in queries:
    print(f"比較中: {q} …", file=sys.stderr)
    block = compare(q)
    blocks.append(block)
    print(block)  # 標準出力にも流す

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write("\n\n".join(blocks) + "\n")
print(f"\n比較結果を {OUT_PATH} に保存しました。", file=sys.stderr)
