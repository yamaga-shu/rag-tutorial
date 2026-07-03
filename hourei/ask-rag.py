import os, sys

# リポジトリ直下を import パスに追加（共通モジュール rag_common を参照するため）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from rag_common import build_hybrid_retriever, get_llm

# Step1. Retriever を用意（密ベクトル + BM25 を RRF 融合したハイブリッド、上位4件）
#   wikipedia 例では密のみ retriever だったが、ここをハイブリッドに差し替えるだけで
#   下流の RAG チェーンはそのまま使える。
retriever = build_hybrid_retriever('hourei/hourei.db', 'hourei/hourei_passages.jsonl', k = 4)

# Step2. LLM と、検索した文脈を渡すためのプロンプトを用意
llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ('system',
     'あなたは提示された条文だけを根拠に日本語で答える法令アシスタントです。'
     '答える際は根拠にした条番号を必ず示してください。'
     '条文に答えが無い場合は「分かりません」と答えてください。\n\n'
     '条文:\n{context}'),
    ('human', '{input}'),
])

# Step3. RAG チェーンを組み立てる
#   create_stuff_documents_chain : 検索した条文を {context} に詰めて LLM に渡す
#   create_retrieval_chain       : retriever と上記チェーンを繋ぐ
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Step4. コマンドライン引数、なければ既定のクエリで質問
query = sys.argv[1] if len(sys.argv) > 1 else "私的使用のための複製はどこまで認められますか？"

result = rag_chain.invoke({'input': query})

print(f"質問: {query}\n")
print(f"回答:\n{result['answer']}\n")
print("-" * 40)
print("参照した条文:")
for i, doc in enumerate(result['context'], 1):
    m = doc.metadata
    # 出典は「条番号 見出し 項番号」まで示す（項ごとに分けたときは項番号が入る）。
    cite = ' '.join(x for x in [m.get('law', ''), m.get('article', ''),
                                m.get('caption', ''), m.get('paragraph', '')] if x)
    print(f"[{i}] 出典: {cite} ({m.get('source', '')})")
    print(doc.page_content[:80].replace("\n", "") + " …")
