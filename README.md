# rag-tutorial

個人的な RAG 構築のためのチュートリアルリポジトリ。

分割 → e5 で埋め込み → FAISS → retriever → Claude という RAG の骨格は共通で、データ源（青空文庫の文学作品と Wikipedia 記事）だけが異なる2つの例を収録している。
さらに、密ベクトル検索に **BM25**（日本語のキーワード検索）を組み合わせたハイブリッド検索の例（著作権法）も収録している。

## セットアップ

```bash
uv sync                       # 依存関係のインストール
cp .env.example .env          # Anthropic API キーを .env に設定
```

以降のコマンドはすべてリポジトリ直下から実行する。

## joseito（青空文庫『女生徒』を題材にした RAG）

太宰治『女生徒』の本文を題材に、質問応答を行う。

```bash
uv run python joseito/get-text-from-url.py       # 本文を取得 → joseito/joseito.txt
uv run python joseito/mk-rag-db-from-text.py     # チャンク分割と埋め込み → joseito/joseito.db
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

検索語に関連する Wikipedia 記事群を取得し、質問応答を行う。
取得段階で記事名と URL をメタデータとして保持し、回答に出典を添える。

```bash
uv run python wikipedia/get-articles-from-wikipedia.py "深層学習" 3   # 上位3記事を取得 → wikipedia/wikipedia.jsonl
uv run python wikipedia/mk-rag-db-from-text.py                        # チャンク分割と埋め込み → wikipedia/wikipedia.db
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

## hourei（著作権法を題材にしたハイブリッド検索 RAG）

密ベクトル検索（e5 + FAISS）と **BM25**（キーワードによる字句一致）を組み合わせ、両者のランキングを **RRF**（Reciprocal Rank Fusion）で融合したハイブリッド検索の例。
「私的使用のための複製」「職務著作」のような法律用語の完全一致が効く場面で、密ベクトル単独では取りこぼす条文を BM25 が拾い上げる様子を確認できる。

```bash
uv run python hourei/get-laws-from-egov.py                    # e-Gov 法令API から著作権法を取得 → hourei/hourei.jsonl（1行=1項、短い条は1条）
uv run python hourei/mk-rag-db-from-text.py                   # チャンク整形と埋め込み → hourei/hourei.db（密）+ hourei_passages.jsonl（BM25用）
uv run python hourei/search-rag-db.py "私的使用のための複製"    # 密のみ / BM25のみ / ハイブリッド の3通りを並べて比較
uv run python hourei/ask-rag.py "私的使用のための複製はどこまで認められますか？"   # ハイブリッド検索で Claude 質問応答（出典＝条番号付き）
```

`get-laws-from-egov.py` の引数は法令ID（省略時は著作権法 `345AC0000000048`）。

日本語の BM25 は分かち書き（トークナイズ）の品質が肝になるため、`rag_common.py` の `tokenize_ja` で **Sudachi** を使って単語分割してから索引化している。
密ベクトルの FAISS は永続化する一方、BM25 索引は軽いので `hourei_passages.jsonl`（分割済みチャンク）から検索時に組み立て直す。

### 密のみとハイブリッドで回答はどう変わるか

`ask-rag.py` は常にハイブリッド検索を使う。
密ベクトルだけの検索（joseito や wikipedia と同じ構成）と比べて回答がどう変わるかは、`compare-ask.py` で確かめられる。
同じ質問を密のみ／ハイブリッドの両方で問い、Claude の回答と参照した条文を並べて `hourei/compare/ask-compare.txt` に書き出す。

```bash
uv run python hourei/compare-ask.py "図書館では著作物をどこまで複製できますか？"   # 密のみ vs ハイブリッド → hourei/compare/ask-compare.txt
```

「図書館では著作物をどこまで複製できますか？」を両方で問うと、参照した条文にはっきり差が見られた。

| 検索 | 拾った第三十一条の項 | 混じった別条 |
|---|---|---|
| 密ベクトルのみ | 第1、4、7項 | 第四十二条の四 |
| ハイブリッド | 第1、2、4、5、7項 | 第四十二条の四 |

第三十一条（図書館等における複製等）は、一部分の複製、保存、絶版等資料の提供、公衆送信と、複製の場面ごとに項が分かれている。
密ベクトルのみでは第1、4、7項までしか届かず、残る枠を別条の第四十二条の四（公文書管理法等による保存等のための利用）が占める。
ハイブリッドは密と BM25 の2つのランキングを RRF で統合するため候補が広がり、その広がった枠を「図書館」「複製」の字句一致で引き寄せた同じ第三十一条の項が埋める。
これにより、密のみが取りこぼした第2項（特定図書館等による公衆送信の規定）と第5項も拾う。

その結果、回答も第三十一条の枠内で複製の場面をより広く押さえたものになる。
両方の回答全文は `hourei/compare/ask-compare.txt` に残る。

> e-Gov 法令API v2（`https://laws.e-gov.go.jp/api/2/law_data/{法令ID}?response_format=json`）を利用している。
> 本文は `{tag, attr, children}` の入れ子ツリーで返り、条は項（Paragraph）、項は号（Item）に分かれている。
> スクリプト側でこの構造をたどり、短い条は条まるごと、長い条は項ごとに切り出して、条番号、見出し、項番号、本文を取り出している。

## 参考
- [LLMのファインチューニングとRAG ―チャットボット開発による実践―](https://www.amazon.co.jp/dp/B0D46V4B9W)
