# rag-tutorial

個人的な RAG 構築のためのチュートリアルリポジトリ。

分割 → e5 で埋め込み → FAISS → retriever → Claude という RAG の骨格は共通で、
データ源（青空文庫の文学作品 / Wikipedia 記事）だけが異なる2つの例を収録している。

## セットアップ

```bash
uv sync                       # 依存関係のインストール
cp .env.example .env          # Anthropic API キーを .env に設定
```

以降のコマンドはすべて **リポジトリ直下** から実行する。

## joseito（青空文庫『女生徒』を題材にした RAG）

太宰治『女生徒』の本文を題材に、質問応答を行う。

```bash
uv run python joseito/get-text-from-url.py       # 本文を取得 → joseito/joseito.txt
uv run python joseito/mk-rag-db-from-text.py     # チャンク分割・埋め込み → joseito/joseito.db
uv run python joseito/search-rag-db.py "朝起きたときの気持ち"   # 類似検索
uv run python joseito/ask-rag.py "朝起きたときの気持ちは？"     # Claude で質問応答
```

出力例（`ask-rag.py`）:

```
質問: 朝起きたときの気持ちは？

回答:
朝起きたときの気持ちは、非常に憂鬱で否定的なものとして描かれています。「朝は灰色」
「一ばん虚無だ」と表現され、疲れて何もしたくない、厭世的な気分だと述べられています。
----------------------------------------
参照した文脈:
[1] に浮かんで、やりきれない。いやだ。いやだ。朝の私は一ばん醜い。…
[2] 同じ。一ばん虚無だ。朝の寝床の中で、私はいつも厭世的だ。…
```

## wikipedia（Wikipedia 記事を題材にした RAG）

検索語に関連する Wikipedia 記事群を取得し、質問応答を行う。取得段階で記事名・URL を
メタデータとして保持し、回答に出典を添える。

```bash
uv run python wikipedia/get-articles-from-wikipedia.py "深層学習" 3   # 上位3記事を取得 → wikipedia/wikipedia.jsonl
uv run python wikipedia/mk-rag-db-from-text.py                        # チャンク分割・埋め込み → wikipedia/wikipedia.db
uv run python wikipedia/search-rag-db.py "強化学習とは"              # 類似検索（出典付き）
uv run python wikipedia/ask-rag.py "強化学習とは何ですか？"          # Claude で質問応答（出典付き）
```

`get-articles-from-wikipedia.py` の引数は「検索語」と「取得記事数（省略時5）」。

出力例（`ask-rag.py`）:

```
質問: 強化学習とは何ですか？

回答:
強化学習（英: reinforcement learning、RL）とは、ある環境内における知的エージェントが、
現在の状態を観測し、得られる収益（累積報酬）を最大化するために、どのような行動をとる
べきかを決定する機械学習の一分野です。教師あり学習、教師なし学習と並ぶ3つの基本的な
機械学習パラダイムの一つとされています。
----------------------------------------
参照した文脈:
[1] 出典: 強化学習 (https://ja.wikipedia.org/wiki/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92)
強化学習（きょうかがくしゅう、英: reinforcement learning、RL）は、…
[2] 出典: 強化学習 (https://ja.wikipedia.org/wiki/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92)
強化学習はその一般性から、ゲーム理論、制御理論、…
```

> Wikimedia API は既定の User-Agent を 429（レート制限）で拒否するため、
> `get-articles-from-wikipedia.py` 内で `wikipedia.set_user_agent(...)` を設定している。

## 参考
- [LLMのファインチューニングとRAG ―チャットボット開発による実践―](https://www.amazon.co.jp/dp/B0D46V4B9W)
