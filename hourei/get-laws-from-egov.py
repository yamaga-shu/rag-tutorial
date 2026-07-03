import json
import sys
import urllib.request

# e-Gov 法令API v2 から法令本文(JSON)を取得し、項を単位にした JSONL にする。
#   API 仕様: https://laws.e-gov.go.jp/api/2/  （response_format=json でツリー構造の本文が返る）
# 著作権法 = 法令ID 345AC0000000048（昭和四十五年法律第四十八号）を既定にする。
law_id = sys.argv[1] if len(sys.argv) > 1 else '345AC0000000048'

# 条本文がこの文字数を超えたら項ごとに分ける。短い条は条まるごと1レコードにする。
SPLIT_THRESHOLD = 500

API = f'https://laws.e-gov.go.jp/api/2/law_data/{law_id}?response_format=json'
# 既定 UA だとレート制限に掛かりやすいので、説明的な UA を付ける（wikipedia 例と同じ配慮）。
req = urllib.request.Request(API, headers = {'User-Agent': 'rag-tutorial/0.1 (RAG tutorial)'})
with urllib.request.urlopen(req, timeout = 60) as res:
    data = json.load(res)

law_title = data['revision_info']['law_title']              # 例: 著作権法
law_url = f'https://laws.e-gov.go.jp/law/{law_id}'           # 法令ページ URL


def collect_text(node):
    """{tag, attr, children} のツリーから、末端の文字列を全て拾って連結する。"""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return ''.join(collect_text(c) for c in node)
    if isinstance(node, dict):
        return ''.join(collect_text(c) for c in node.get('children', []))
    return ''


def find_child(node, tag):
    """node の直下の子から、最初に tag が一致する子を返す（無ければ None）。"""
    for c in node.get('children', []):
        if isinstance(c, dict) and c.get('tag') == tag:
            return c
    return None


def find_children(node, tag):
    """node の直下の子から、tag が一致する子を全て返す。"""
    return [c for c in node.get('children', [])
            if isinstance(c, dict) and c.get('tag') == tag]


def iter_articles(node):
    """ツリーを再帰的に辿り、Article タグのノードを順に返す。"""
    if isinstance(node, dict):
        if node.get('tag') == 'Article':
            yield node
            return  # Article は入れ子にならないので下は辿らない
        for c in node.get('children', []):
            yield from iter_articles(c)
    elif isinstance(node, list):
        for c in node:
            yield from iter_articles(c)


def make_record(article_no, caption, paragraph, body):
    """1レコード分の dict を作る。本文の先頭に見出し行を前置する。

    見出し行は「法令名 条番号 見出し（＋項番号）」。項の途中だけを切り出した
    チャンクでも、単独で条と主題が読み取れ、字句一致(BM25)に見出し語が残る。
    """
    heading = ' '.join(x for x in [law_title, article_no, caption, paragraph] if x)
    return {
        'text': f'{heading}\n{body}',
        'metadata': {
            'law': law_title,        # 法令名
            'article': article_no,   # 条番号（出典表示に使う）
            'caption': caption,      # 見出し
            'paragraph': paragraph,  # 項番号（項ごとに分けたときだけ入る）
            'source': law_url,       # 法令ページ URL
        },
    }


# 本文ツリーから条を1つずつ取り出し、条の長さに応じて条または項の単位で書き出す。
outfile = 'hourei/hourei.jsonl'
articles = 0
records = 0
with open(outfile, 'w', encoding='utf-8') as f:
    for art in iter_articles(data['law_full_text']):
        articles += 1
        title_node = find_child(art, 'ArticleTitle')      # 例: 第三十条
        caption_node = find_child(art, 'ArticleCaption')   # 例: （私的使用のための複製）
        article_no = collect_text(title_node) if title_node else ''
        caption = collect_text(caption_node) if caption_node else ''
        paragraphs = find_children(art, 'Paragraph')       # 項（号は各項の中に含まれる）

        # 短い条・単項の条は条まるごと1レコード。長い条だけ項ごとに分ける。
        # 号は属する項の中に残るので、意味の単位が文の途中で割れない。
        if len(collect_text(art)) <= SPLIT_THRESHOLD or len(paragraphs) <= 1:
            body = ''.join(collect_text(p) for p in paragraphs)
            f.write(json.dumps(make_record(article_no, caption, '', body), ensure_ascii=False) + '\n')
            records += 1
        else:
            for p in paragraphs:
                num = p.get('attr', {}).get('Num', '')     # 項番号（算用数字）
                paragraph = f'第{num}項' if num else ''
                f.write(json.dumps(make_record(article_no, caption, paragraph, collect_text(p)), ensure_ascii=False) + '\n')
                records += 1

print(f'「{law_title}」から {articles} 条を {records} レコード（項単位）で {outfile} に出力しました。')
