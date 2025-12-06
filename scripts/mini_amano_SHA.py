import random
import numpy as np
import json   # ← 追加しておくと便利
import random
random.seed(42)

# =======================================
# ここに JSON をロードする
# =======================================
with open("vocab_build_v1_1.json", "r", encoding="utf-8") as f:
    VOCAB_CLUSTERS = json.load(f)

# =======================================
#  Layer 0：ビル型語彙クラスタ
# =======================================

VOCAB_CLUSTERS = {
    "構造系": [
        "構造","層","形式","位相","軸","密度","配列","節点","対称","射程"
    ],
    "関係系": [
        "関係","連続","反転","遷移","分岐","接続","収束","ゆらぎ","勾配","境界"
    ],
    "象徴系": [
        "象徴","位相差","背景","前景","重心","反響","遠近","影響圏","包絡","透過"
    ],
    "抽象系": [
        "抽象度","解像度","視座","観測","変容","生成","分節","参照","射影","内包"
    ],
    "変動系": [
        "多相","変換","拡張","再編","反照","振動","環流","帯域","記号化","誘導"
    ]
}

# 平坦化して ID を振り直す
FLATTEN_VOCAB = []
CLUSTER_INDEX = {}  # cluster名 → (start, end)

start = 0
for cluster, words in VOCAB_CLUSTERS.items():
    FLATTEN_VOCAB.extend(words)
    end = start + len(words)
    CLUSTER_INDEX[cluster] = (start, end)
    start = end

word_to_id = {w: i for i, w in enumerate(FLATTEN_VOCAB)}
id_to_word = {i: w for i, w in enumerate(FLATTEN_VOCAB)}

# ベクトル次元
EMBED_DIM = 12

# ビル型の意味ベクトル（クラスタ単位で中心が異なる）
np.random.seed(42)
cluster_centers = {c: np.random.randn(EMBED_DIM) for c in VOCAB_CLUSTERS}

embedding = np.zeros((len(FLATTEN_VOCAB), EMBED_DIM))
for cluster, (s, e) in CLUSTER_INDEX.items():
    center = cluster_centers[cluster]
    for i in range(s, e):
        embedding[i] = center + np.random.randn(EMBED_DIM) * 0.2


# =======================================
# 文章生成（修正版）
#  - clusters の数 < length の場合でも動くように重複選択を許可
# =======================================
def generate_sentence(length=5):
    clusters = list(VOCAB_CLUSTERS.keys())
    # random.choices で重複を許可してクラスタを選ぶ（母集団より多くてもOK）
    selected_clusters = random.choices(clusters, k=length)

    sentence = []
    for c in selected_clusters:
        words = VOCAB_CLUSTERS[c]
        sentence.append(random.choice(words))

    return sentence


# =======================================
# 文章 → ベクトル
# =======================================
def sentence_to_vector(words):
    vecs = [embedding[word_to_id[w]] for w in words]
    return np.mean(vecs, axis=0)


# =======================================
# ベクトル → 語彙（クラスタ優先で近い語句に戻す）
# =======================================
def vector_to_sentence(vec, length=5):
    sims = []

    for i, e in enumerate(embedding):
        sim = np.dot(vec, e) / (np.linalg.norm(vec) * np.linalg.norm(e))
        sims.append((sim, i))

    sims.sort(reverse=True)
    top_ids = [i for _, i in sims[:length]]

    return [id_to_word[i] for i in top_ids]


# =======================================
# Self-Poly（自己多相）ループ ＋ 軌跡ログ
# =======================================
def self_poly(iterations=3, length=5):
    sentence = generate_sentence(length)
    print("起点:", sentence)

    # ← 追加：軌跡ログ（文章ベクトルの履歴）
    trajectory = []

    # 起点ベクトルも記録
    vec0 = sentence_to_vector(sentence)
    trajectory.append(vec0)

    for i in range(iterations):
        vec = sentence_to_vector(sentence)
        sentence = vector_to_sentence(vec, length)
        print(f"第{i+1}多相:", sentence)

        # ← 毎ステップのベクトルを記録
        trajectory.append(vec)

    return sentence, trajectory


# =======================================
# 実行
# =======================================
if __name__ == "__main__":
    final, trajectory = self_poly(iterations=6, length=7)

import json

with open("final_output.json", "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

    print("\n最終生成:", final)

    # ← ベクトル軌跡を保存
    np.save("trajectory.npy", np.array(trajectory))
    print("軌跡を trajectory.npy として保存しました。")
