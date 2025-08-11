import torch
# 目的: 自己注意で「どのトークンにどれだけ注目するか」を学習できるよう、
# 各トークンを Q/K/V に写像する土台を作る最小例

# 入力: 各行が1トークンの埋め込みベクトル
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 目的: 2番目のトークンをクエリとして使い、関連度計算の基点にする
x_2 = inputs[1]
d_in = inputs.shape[1] # 入力埋め込みのサイズ
d_out = 2 # 出力埋め込みのサイズ

torch.manual_seed(123)  # 目的: 実験の再現性（結果比較・検証のため）
# 目的: Q/K/V を別役割で表現できるようにする写像
#  - Q: 何に注目するか（問い合わせの視点）
#  - K: どれだけ注目されるか（鍵/一致度の指標）
#  - V: 渡す情報そのもの
#   ※ 本来は学習するが、ここでは固定して挙動を確認
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query  # 目的: このトークンが「何に注目するか」を表す
key_2 = x_2 @ W_key      # 目的: このトークンが「どれだけ注目されるか」の指標
value_2 = x_2 @ W_value  # 目的: このトークンが「相手へ渡す情報」

keys = inputs @ W_key 
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)