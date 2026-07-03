with open('joseito.txt', 'r', encoding='utf-8') as f:
    text = f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100, #チャンクの文字数
    chunk_overlap = 0, #チャンクオーバーラップの文字数
)

texts = text_splitter.split_text(text) 
