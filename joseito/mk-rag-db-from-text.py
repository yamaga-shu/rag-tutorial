# Step1. パッセージへの分解
with open('joseito/joseito.txt', 'r', encoding='utf-8') as f:
    text = f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100, #チャンクの文字数
    chunk_overlap = 0, #チャンクオーバーラップの文字数
)

passages = text_splitter.split_text(text) 

# Step2. パッセージのベクトル化
embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs = {'device':'mps'}
)

# Step3. DBの作成
db = FAISS.from_texts(passages, embeddings)
db.save_local('joseito/joseito.db')
