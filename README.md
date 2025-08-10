# LLM-Tukuttemita

https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences


<details>
<summary>HandmadeTokenizer.py</summary>

### 概要
テキストを機械学習で扱える数値（トークンID）に変換する、シンプルなトークナイザーの実装です。

### データ処理の流れ

1. **サンプルテキストの準備**
   - 「The Verdict」という短編小説をダウンロード（自動）
   - 約5,000単語のテキストデータ

2. **トークン化処理**
   - 正規表現 `r'([,.:;?_!"()\']|--|\s)'` でテキストを分割
   - 単語と句読点を個別のトークンとして扱う
   - 例: "Hello, world!" → ['Hello', ',', 'world', '!']

3. **語彙辞書の構築**
   - テキスト内の全ユニークトークンを収集（1132個）
   - 各トークンに一意のIDを割り当て
   - 特殊トークンの追加:
     - `<|endoftext|>`: 文章の終端マーカー（ID: 1130）
     - `<|unk|>`: 語彙にない単語用（ID: 1131）

4. **エンコード・デコード処理**
   - エンコード: テキスト → トークン分割 → ID変換
   - デコード: ID列 → トークン復元 → テキスト結合

### 実際の処理例

```python
# エンコード処理
text = "Hello, do you like tea?"
ids = tokenizer.encode(text)
# 処理の流れ:
# 1. トークン分割: ['Hello', ',', 'do', 'you', 'like', 'tea', '?']
# 2. 語彙チェック: Hello → <|unk|>, tea → <|unk|>
# 3. ID変換: [1131, 5, 355, 1126, 628, 1131, 10]

# デコード処理
decoded = tokenizer.decode(ids)
# 処理の流れ:
# 1. ID→トークン: [1131, 5, 355, 1126, 628, 1131, 10]
#                → ['<|unk|>', ',', 'do', 'you', 'like', '<|unk|>', '?']
# 2. 結合・整形: "<|unk|>, do you like <|unk|>?"
```

### データ変換の詳細図

```mermaid
%%{init: {'theme':'dark'}}%%
graph LR
    subgraph "エンコード処理（実際の値）"
        A1["入力テキスト<br/>Hello, do you like tea? endoftext In the sunlit terraces of the palace."] --> A2["正規表現分割<br/>Hello<br/>カンマ<br/>do<br/>you<br/>like<br/>tea<br/>クエスチョン<br/>endoftext<br/>In<br/>the<br/>sunlit<br/>terraces<br/>of<br/>the<br/>palace<br/>ピリオド"]
        A2 --> A3["語彙チェック（実際の結果）<br/>Hello → 未知トークン<br/>カンマ → 語彙にある<br/>do → 語彙にある<br/>you → 語彙にある<br/>like → 語彙にある<br/>tea → 語彙にある<br/>クエスチョン → 語彙にある<br/>endoftext → 語彙にある<br/>In → 語彙にある<br/>the → 語彙にある<br/>sunlit → 語彙にある<br/>terraces → 語彙にある<br/>of → 語彙にある<br/>the → 語彙にある<br/>palace → 未知トークン<br/>ピリオド → 語彙にある"]
        A3 --> A4["未知トークン置換<br/>unk<br/>カンマ<br/>do<br/>you<br/>like<br/>tea<br/>クエスチョン<br/>endoftext<br/>In<br/>the<br/>sunlit<br/>terraces<br/>of<br/>the<br/>unk<br/>ピリオド"]
        A4 --> A5["ID変換（実際の値）<br/>1131<br/>5<br/>355<br/>1126<br/>628<br/>975<br/>10<br/>1130<br/>55<br/>988<br/>956<br/>984<br/>722<br/>988<br/>1131<br/>7"]
    end
    
    subgraph "デコード処理（実際の値）"
        B1["入力ID<br/>1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7"] --> B2["ID→トークン変換<br/>unk<br/>カンマ<br/>do<br/>you<br/>like<br/>tea<br/>クエスチョン<br/>endoftext<br/>In<br/>the<br/>sunlit<br/>terraces<br/>of<br/>the<br/>unk<br/>ピリオド"]
        B2 --> B3["テキスト結合<br/>unk カンマ do you like tea クエスチョン endoftext In the sunlit terraces of the unk ピリオド"]
        B3 --> B4["句読点整形<br/>unk, do you like tea? endoftext In the sunlit terraces of the unk."]
    end
    
    A5 --> B1
    
    style A1 fill:#374151,color:#ffffff
    style A2 fill:#374151,color:#ffffff
    style A3 fill:#374151,color:#ffffff
    style A4 fill:#374151,color:#ffffff
    style A5 fill:#374151,color:#ffffff
    style B1 fill:#374151,color:#ffffff
    style B2 fill:#374151,color:#ffffff
    style B3 fill:#374151,color:#ffffff
    style B4 fill:#374151,color:#ffffff
```

### ポイント
- 教育目的のシンプルな実装で、トークン化の基本原理を理解できる
- 語彙サイズが小さい（1132個）ため、多くの単語が未知語になる
- 実用的なトークナイザー（BPE、WordPiece等）の基礎となる概念

</details>

<details>
<summary>bytePairEncorder.py</summary>

### 概要
GPT-2のトークナイザー（tiktoken）を使用して、LLM学習用のデータを準備するファイルです。テキストを固定長のシーケンスに分割し、次の単語を予測する学習データを作成します。

### データ処理の流れ

1. **テキストのトークン化**
   - GPT-2のBPE（Byte Pair Encoding）トークナイザーを使用
   - 語彙サイズ: 50,257トークン
   - 例: "I HAD always" → [40, 367, 2885]

2. **スライディングウィンドウによるシーケンス生成**
   - `max_length`: シーケンスの長さ（例: 4）
   - `stride`: ウィンドウの移動幅（例: 1）
   - オーバーラップにより、データを最大限活用

3. **入力と目的変数のペア作成**
   - 入力: 現在のトークン列
   - 目的変数: 1つずつ右にシフトしたトークン列（次の単語を予測）
   ```
   入力:  [40, 367, 2885, 1464]
   目的:  [367, 2885, 1464, 1807]
   ```

4. **バッチ処理の準備**
   - 複数のシーケンスをまとめて処理
   - GPUでの並列計算に最適化

### 実際のデータ処理例

```python
# パラメータ設定
context_size = 4  # シーケンス長
stride = 1        # 1トークンずつシフト
batch_size = 8    # 8個のシーケンスを同時処理

# データの流れ
# 1. テキスト: "I HAD always thought Jack Gisburn..."
# 2. トークンID: [40, 367, 2885, 1464, 1807, 3619, ...]
# 3. シーケンス分割:
#    - seq1: [40, 367, 2885, 1464] → [367, 2885, 1464, 1807]
#    - seq2: [367, 2885, 1464, 1807] → [2885, 1464, 1807, 3619]
#    - seq3: [2885, 1464, 1807, 3619] → [1464, 1807, 3619, 402]
```

### データ変換の詳細図

```mermaid
%%{init: {'theme':'dark'}}%%
flowchart LR
 subgraph s1["元のテキスト"]
        A1@{ label: "'I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow e...'" }
  end
 subgraph s2["トークン化"]
        B1["[40, 367, 2885, 1464, 1807, 3619, 402, 271, 10899, 2138, ...]"]
  end
 subgraph subGraph2["スライディングウィンドウ"]
        C1@{ label: "シーケンス1:<br>[40, 367, 2885, 1464]<br>'I HAD always'" }
        C2@{ label: "シーケンス2:<br>[367, 2885, 1464, 1807]<br>' HAD always thought'" }
        C3@{ label: "シーケンス3:<br>[2885, 1464, 1807, 3619]<br>'AD always thought Jack'" }
  end
 subgraph s3["入力変数と目的変数のペア"]
        D1@{ label: "入力: [40, 367, 2885, 1464] → 'I HAD always'<br>目的: [367, 2885, 1464, 1807] → ' HAD always thought'" }
        D2@{ label: "入力: [367, 2885, 1464, 1807] → ' HAD always thought'<br>目的: [2885, 1464, 1807, 3619] → 'AD always thought Jack'" }
        D3@{ label: "入力: [2885, 1464, 1807, 3619] → 'AD always thought Jack'<br>目的: [1464, 1807, 3619, 402] → ' always thought Jack G'" }
  end
 subgraph subGraph4["バッチ処理 (batch_size=8, stride=4)"]
        E1["バッチ1:<br>入力: [[40,367,2885,1464], [1807,3619,402,271], ...]<br>目的: [[367,2885,1464,1807], [3619,402,271,10899], ...]"]
        E2["8個のシーケンスを<br>まとめて処理"]
  end
    A1 --> B1
    B1 --> C1 & C2 & C3
    C1 --> D1
    C2 --> D2
    C3 --> D3
    D1 --> E1
    D2 --> E1
    D3 --> E1
    E1 --> E2

    A1@{ shape: rect}
    C1@{ shape: rect}
    C2@{ shape: rect}
    C3@{ shape: rect}
    D1@{ shape: rect}
    D2@{ shape: rect}
    D3@{ shape: rect}
    style A1 fill:#374151,color:#ffffff
    style B1 fill:#374151,color:#ffffff
    style C1 fill:#374151,color:#ffffff
    style C2 fill:#374151,color:#ffffff
    style C3 fill:#374151,color:#ffffff
    style D1 fill:#374151,color:#ffffff
    style D2 fill:#374151,color:#ffffff
    style D3 fill:#374151,color:#ffffff
    style E1 fill:#374151,color:#ffffff
    style E2 fill:#374151,color:#ffffff
```

### 重要なポイント

- **次の単語予測タスク**: LLMの基本的な学習方法
- **スライディングウィンドウ**: 限られたデータを最大限活用
- **バッチ処理**: GPU計算の効率化
- **PyTorchとの統合**: 標準的な深層学習フレームワークに対応

</details>


<details>
<summary>self-attention.py</summary>

このSled-Attentionでの目標は入力シーケンスの各要素に対してコンテキストベクトルを計算すること

</details>