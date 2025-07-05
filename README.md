# LLM-Tukuttemita

https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences


## HandmadeTokenizer.py

LLM用のテキストトークナイザーを自作して、テキストを数値化

### やっていること

1. 学習元のサンプルテキストをダウンロード
- GitHubから短編小説「The Verdict」をダウンロード
- ローカルファイルとして保存（the-verdict.txt）

2. サンプルテキストをトークン化
- 正規表現を使ってテキストを単語や句読点に分割
- 不要な空白を除去してトークンのリストを作成

3. トークンをトークンIDに変換
- 各トークンに一意の数値IDを割り当て
- エンコード機能（テキスト→トークンID）とデコード機能（トークンID→テキスト）を実装

4. 語彙を作成
- 全てのユニークなトークンから語彙辞書を構築
- 特殊トークン（<|endoftext|>、<|unk|>）を追加

5. 自作トークナイザーの動作確認
- サンプルテキストでエンコード・デコード処理を実行