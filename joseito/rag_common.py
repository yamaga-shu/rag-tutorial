from langchain_huggingface import HuggingFaceEmbeddings


class E5Embeddings(HuggingFaceEmbeddings):
    """intfloat/e5 系モデル用。埋め込み時のみ接頭辞を付与する。

    - パッセージ側: "passage: " を付けて埋め込む
    - クエリ側:     "query: "   を付けて埋め込む
    接頭辞は埋め込み計算にのみ使われ、DBに保存される本文は元のまま。
    """

    def embed_documents(self, texts):
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def get_embeddings():
    """DB作成・検索で共通して使う埋め込みモデルを返す。"""
    return E5Embeddings(
        model_name = "intfloat/multilingual-e5-large",
        model_kwargs = {'device': 'mps'},
    )
