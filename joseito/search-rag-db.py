import sys

from langchain_community.vectorstores import FAISS

from rag_common import get_embeddings

# DB作成時と同じ埋め込みモデルを用意
embeddings = get_embeddings()

# 保存したDBを読み込む
db = FAISS.load_local(
    'joseito/joseito.db',
    embeddings,
    allow_dangerous_deserialization = True,  # 自分で作成したDBなので許可
)

# コマンドライン引数、なければ既定のクエリで検索
query = sys.argv[1] if len(sys.argv) > 1 else "朝起きたときの気持ち"

# 類似度スコア付きで上位4件を検索
results = db.similarity_search_with_score(query, k=4)

print(f"検索クエリ: {query}\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"[{i}] score={score:.4f}")
    print(doc.page_content)
    print("-" * 40)
