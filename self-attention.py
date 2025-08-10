import torch

"""
これは「自己注意(Self-Attention)って結局なにをしているの?」を
超ミニ例で体感するためのスクリプトです。

やっていることはシンプル: 2番目の単語をクエリに選び、
他の各単語とどれくらい「似ているか」を内積で測ります。
数値が大きいほど「強く関係している=注目したい」ことを意味します。

注: 本格的な実装でよく見るスケーリング(1/sqrt(d))や softmax による正規化、
    さらに重み付き平均でコンテキストベクトルを作る処理は、
    ここでは理解を優先して省いています。
"""

# 入力: 各行が1トークンの埋め込みベクトル(次元=3の小さなおもちゃデータ)
# ここでは簡単のため、Q=K=V=inputs とみなします。
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# クエリ: 2番目(インデックス1)を採用。ここでは "journey" に対応
query = inputs[1]

# 注意スコア(未正規化)を入れる箱。サイズはトークン数と同じ
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    # 内積: x^i・query → 類似度(注意スコア)として解釈
    attn_scores_2[i] = torch.dot(x_i, query)

# 未正規化スコアを出力
# 参考: 実運用では softmax(attn_scores_2 / sqrt(d)) で正規化して「確率っぽい重み」に変え、
#       その重みで値ベクトル(V)の加重平均を取り、コンテキストベクトルを作ります。
print(attn_scores_2)

# 重みの総和は1になる
# 重みが常に性になる
# 出力を確立(相対的な重要度)として解釈できるようになる
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# 性能面で広く最適化されているPyTorchのソフトマックス
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

query = inputs[1] # 2nd input token is the query

# コンテキストベクトル = 各入力ベクトルと対応するAttentionの重みをかけ合わせて加算した加重和
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)

print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

print("Previous 2nd context vector:", context_vec_2)