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


def load_db(db_path = 'joseito/joseito.db'):
    """保存済みの FAISS DB を読み込んで返す。検索・QA で共通利用する。"""
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(
        db_path,
        get_embeddings(),
        allow_dangerous_deserialization = True,  # 自分で作成したDBなので許可
    )


def get_llm():
    """回答生成に使う Claude を返す。モデル名・パラメータをここに集約する。"""
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic

    load_dotenv()  # .env の ANTHROPIC_API_KEY を読み込む
    return ChatAnthropic(
        model = "claude-sonnet-5",
        max_tokens = 1024,
    )


# ここから下は hourei（法令ハイブリッド検索）の例で使う。
# joseito / wikipedia の密ベクトルだけの例は上の関数のみで動く。

# Sudachi の分かち書き器は生成コストが高いので、最初の呼び出し時に一度だけ作る。
_tokenizer = None


def tokenize_ja(text):
    """日本語文を分かち書きして単語リストにする（BM25 の索引・検索に使う）。

    BM25 は「単語の一致」で効くため、日本語では前段の分かち書き品質が肝になる。
    ここでは Sudachi の Mode C（できるだけ長い単位）を使い、「職務著作」「複製権」の
    ような法律用語をなるべく1語として残す。空白・記号だけのトークンは落とす。
    """
    global _tokenizer
    if _tokenizer is None:
        from sudachipy import dictionary
        _tokenizer = dictionary.Dictionary().create()

    from sudachipy import SplitMode
    words = []
    for m in _tokenizer.tokenize(text, SplitMode.C):
        w = m.surface().strip()
        if w:
            words.append(w)
    return words


def build_hybrid_retriever(db_path, passages_path, k = 4, weights = (0.5, 0.5)):
    """密ベクトル(FAISS) と 疎(BM25) を RRF で融合したハイブリッド retriever を返す。

    - 密: load_db で読み込んだ FAISS。意味の近さに強い。
    - 疎: 保存済みチャンク(JSONL)から毎回組み立てる BM25。字句の完全一致に強い。
          （BM25 索引は軽いので永続化せず、検索時に passages から作り直す）
    - 融合: EnsembleRetriever が両者のランクを RRF（Reciprocal Rank Fusion）で統合する。
    """
    import json

    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    # 密ベクトル側の retriever
    dense = load_db(db_path).as_retriever(search_kwargs = {'k': k})

    # 疎(BM25)側の retriever … 保存しておいたチャンクから復元して索引化
    passages = []
    with open(passages_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            passages.append(Document(page_content = r['text'], metadata = r['metadata']))
    bm25 = BM25Retriever.from_documents(passages, preprocess_func = tokenize_ja)
    bm25.k = k

    return EnsembleRetriever(retrievers = [bm25, dense], weights = list(weights))
