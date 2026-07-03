import json
import sys

import wikipedia
from langchain_community.document_loaders import WikipediaLoader

# Wikimedia API は既定の User-Agent を 429 で拒否するため、説明的な UA を設定する。
#   参考: https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits
# WikipediaLoader は内部でこの wikipedia パッケージを使うので、読み込み前に設定すれば効く。
wikipedia.set_user_agent("rag-tutorial/0.1 (RAG tutorial; contact via repository)")

# コマンドライン引数、なければ既定の検索語で取得
query = sys.argv[1] if len(sys.argv) > 1 else "深層学習"
max_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# 検索語に関連する上位 max_docs 件の日本語記事を取得
#   各 Document には title / source(URL) などのメタデータが入る
loader = WikipediaLoader(query=query, lang="ja", load_max_docs=max_docs)
docs = loader.load()

# 記事ごとに1行の JSON Lines で保存（メタデータを失わないため）
outfile = 'wikipedia/wikipedia.jsonl'
with open(outfile, 'w', encoding='utf-8') as f:
    for doc in docs:
        record = {
            'title': doc.metadata.get('title', ''),
            'source': doc.metadata.get('source', ''),
            'text': doc.page_content,
        }
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"検索語「{query}」で {len(docs)} 件の記事を {outfile} に出力しました。")
for doc in docs:
    print(f"  - {doc.metadata.get('title', '')}")
