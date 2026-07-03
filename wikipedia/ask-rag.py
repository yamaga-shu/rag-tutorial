import os, sys

# リポジトリ直下を import パスに追加（共通モジュール rag_common を参照するため）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from rag_common import load_db, get_llm

# Step1. Retriever を用意（DBから上位2件を取得）
retriever = load_db('wikipedia/wikipedia.db').as_retriever(search_kwargs = {'k': 2})

# Step2. LLM と、検索した文脈を渡すためのプロンプトを用意
llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ('system',
     'あなたは提示された文脈だけを根拠に日本語で答えるアシスタントです。'
     '文脈に答えが無い場合は「分かりません」と答えてください。\n\n'
     '文脈:\n{context}'),
    ('human', '{input}'),
])

# Step3. RAG チェーンを組み立てる
#   create_stuff_documents_chain : 検索した文書を {context} に詰めて LLM に渡す
#   create_retrieval_chain       : retriever と上記チェーンを繋ぐ
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Step4. コマンドライン引数、なければ既定のクエリで質問
query = sys.argv[1] if len(sys.argv) > 1 else "ニューラルネットワークとは何ですか？"

result = rag_chain.invoke({'input': query})

print(f"質問: {query}\n")
print(f"回答:\n{result['answer']}\n")
print("-" * 40)
print("参照した文脈:")
for i, doc in enumerate(result['context'], 1):
    title = doc.metadata.get('title', '')
    source = doc.metadata.get('source', '')
    print(f"[{i}] 出典: {title} ({source})")
    print(doc.page_content)
