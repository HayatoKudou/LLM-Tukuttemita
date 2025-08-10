import torch

# 入力: 各行が1トークンの埋め込みベクトル
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# クエリ: 2つ目の入力トークン「journey」をクエリとして使う
query = inputs[1]

# Attentionスコアの計算
# クエリと他の入力トークンの間でドット積を計算することで決定
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    # 内積: x^i・query → Attentionスコアとして解釈
    attn_scores_2[i] = torch.dot(x_i, query)

# Attentionスコアを正規化
# 正規化の主な目的は、Attentionの重みの総和が1になるようにして、、確率的な解釈ができるように
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

# 実際には、より数値的に安定した方法で正規化する
# 性能面で広く最適化されているPyTorchのソフトマックス
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

# コンテキストベクトル = 各入力ベクトルに対応する注意重みを掛けて足し合わせた加重和
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)


# ↓ より洗練した実装

# forループは遅いので、効率的にすべてのクエリに対して一度にスコアを計算
attn_scores = inputs @ inputs.T

# 各行を正規化
attn_weights = torch.softmax(attn_scores, dim=-1)

# これらAttentionの重みと行列積を使って、すべてのクエリのコンテキストベクトルを計算
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)