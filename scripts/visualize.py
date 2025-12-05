import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --------- 軌跡データ読み込み ----------
trajectory = np.load("trajectory.npy")   # shape: (steps, 12)

# --------- PCAで2次元へ ----------
pca = PCA(n_components=2)
points_2d = pca.fit_transform(trajectory)

# --------- プロット ----------
plt.figure(figsize=(6, 6))

# 軌跡（線）
plt.plot(points_2d[:, 0], points_2d[:, 1], marker='o')

# 起点（青）
plt.scatter(points_2d[0, 0], points_2d[0, 1], s=120, label="Start", edgecolors='black')

# 最終点（赤）
plt.scatter(points_2d[-1, 0], points_2d[-1, 1], s=120, label="End", edgecolors='black')

plt.title("Self-Poly Trajectory (Amano Mini)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
