import requests
from bs4 import BeautifulSoup

url = 'https://www.aozora.gr.jp/cards/000035/files/275_13903.html'

#URL からページの内容を取得
response = requests.get(url)

#HTML の解析
soup = BeautifulSoup(response.content, 'html.parser')

#不要なタグを除去
for script_or_style in soup(['script', 'style']):
    script_or_style.extract()
    
#すべてのテキストを抽出
text = soup.get_text()

#行を分割して、先頭と末尾の空白を削除
lines = (line.strip() for line in text.splitlines())

#空の行を除去
chunks = (phrase.strip() for line in lines for phrase in line.split())

#テキストを一つの文字列に結合
text = '\n'.join(chunk for chunk in chunks if chunk)

#テキストをファイルに保存
outtext = 'joseito.txt'
with open(outtext, 'w', encoding='utf-8') as file:  file.write(text)

print(url, "の内容を", outtext, "に出力しました。")
