import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# バッチ入力と目的変数のためのデータセット
# PyTorch の Dataset クラスをベースとし、データセットから個々の行をどのように取り出すか定義
# データセットの各行は max_lengthの長さのトークンIDで構成され、input_chunkテンソルに代入される
# target_chunkテンソルには、対応する目的変数が代入される
# バッチ入力: 複数のデータをまとめて一度に処理すること
# データセット： 機械学習で使う学習用のデータの集まり
# テンソル: 多次元配列を表現するデータ構造
# シーケンス: 順番に並んだデータの列
class GPTdatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) # テキスト全体をトークン化

        # スライドウィンドを使って 「The Verdict」 をmax_lengthの長さのシーケンスに分割
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): # データセットに含まれている行の総数を返す
        return len(self.input_ids)
    
    def __getitem__(self, idx): # データセットから1行を返す
        return self.input_ids[idx], self.target_ids[idx]
    
context_size = 4
stride = 1
dataset = GPTdatasetV1(raw_text, tokenizer, context_size, stride)

# 入力変数と目的変数のペアでバッチを生成するデータローダー
# batch_size: 一度に処理するデータの個数
# shuffle: データをランダムに並べ替え学習の偏りを防ぐか
# drop_last: データの個数がbatch_sizeの倍数にならない場合、最後のバッチを捨てるか(不完全なバッチを捨てる)
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2") # トークナイザーを初期化
    dataset = GPTdatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers # 前処理に使うCPUプロセスの数
    )
    return dataloader


dataloader = create_dataloader(
    raw_text, 
    batch_size=8,
    max_length=4, 
    stride=4,
    shuffle=False
)

# データローダーをPythonのイテレータに変換し、Pythonの組み込み関数next()で次のエントリを取得
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs: \n", inputs)
print("\Inputs shape: \n", inputs.shape)