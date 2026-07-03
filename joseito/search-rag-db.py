import os, sys

# リポジトリ直下を import パスに追加（共通モジュール rag_common を参照するため）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_common import load_db

# 保存したDBを読み込む（埋め込みモデルの用意も load_db が担う）
db = load_db()

# コマンドライン引数、なければ既定のクエリで検索
query = sys.argv[1] if len(sys.argv) > 1 else "朝起きたときの気持ち"

# 類似度スコア付きで上位4件を検索
results = db.similarity_search_with_score(query, k=4)

print(f"検索クエリ: {query}\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"[{i}] score={score:.4f}")
    print(doc.page_content)
    print("-" * 40)
