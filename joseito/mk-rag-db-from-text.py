# Step1. パッセージへの分解
with open('joseito/joseito.txt', 'r', encoding='utf-8') as f:
    text = f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from rag_common import get_embeddings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100, #チャンクの文字数
    chunk_overlap = 0, #チャンクオーバーラップの文字数
)

passages = text_splitter.split_text(text) 

# Step2. パッセージのベクトル化
embeddings = get_embeddings()

# Step3. DBの作成
db = FAISS.from_texts(passages, embeddings)
db.save_local('joseito/joseito.db')
