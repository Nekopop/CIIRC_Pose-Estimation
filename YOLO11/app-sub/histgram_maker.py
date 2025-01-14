import matplotlib.pyplot as plt
import numpy as np

# データの読み込み
data = """
Distance 1.0m - Overall Mean error: 4.695m, Overall Std: 0.000m, Total Count: 1
Distance 1.5m - Overall Mean error: 0.737m, Overall Std: 0.069m, Total Count: 4
Distance 2.0m - Overall Mean error: 0.338m, Overall Std: 0.080m, Total Count: 35
Distance 2.5m - Overall Mean error: 0.532m, Overall Std: 0.405m, Total Count: 89
Distance 3.0m - Overall Mean error: 0.367m, Overall Std: 0.107m, Total Count: 133
Distance 3.5m - Overall Mean error: 0.413m, Overall Std: 0.163m, Total Count: 153
Distance 4.0m - Overall Mean error: 0.397m, Overall Std: 0.121m, Total Count: 225
Distance 4.5m - Overall Mean error: 0.577m, Overall Std: 0.369m, Total Count: 258
Distance 5.0m - Overall Mean error: 0.673m, Overall Std: 0.391m, Total Count: 181
Distance 5.5m - Overall Mean error: 0.680m, Overall Std: 0.313m, Total Count: 172
Distance 6.0m - Overall Mean error: 0.755m, Overall Std: 0.304m, Total Count: 214
Distance 6.5m - Overall Mean error: 0.740m, Overall Std: 0.298m, Total Count: 235
Distance 7.0m - Overall Mean error: 0.751m, Overall Std: 0.322m, Total Count: 253
Distance 7.5m - Overall Mean error: 0.793m, Overall Std: 0.523m, Total Count: 328
Distance 8.0m - Overall Mean error: 0.902m, Overall Std: 0.387m, Total Count: 250
Distance 8.5m - Overall Mean error: 0.966m, Overall Std: 0.383m, Total Count: 275
Distance 9.0m - Overall Mean error: 0.974m, Overall Std: 0.351m, Total Count: 291
Distance 9.5m - Overall Mean error: 1.146m, Overall Std: 0.385m, Total Count: 287
Distance 10.0m - Overall Mean error: 1.750m, Overall Std: 0.425m, Total Count: 391
Distance 10.5m - Overall Mean error: 2.166m, Overall Std: 0.536m, Total Count: 282
Distance 11.0m - Overall Mean error: 2.579m, Overall Std: 0.496m, Total Count: 301
Distance 11.5m - Overall Mean error: 2.811m, Overall Std: 0.432m, Total Count: 271
"""

# データのパース
lines = data.strip().split('\n')
distances = []
mean_errors = []
std_devs = []
total_counts = []

for line in lines:
    parts = line.split(' - ')
    distance = float(parts[0].split()[1][:-1])
    mean_error = float(parts[1].split(',')[0].split()[-1][:-1])
    std_dev = float(parts[1].split(',')[1].split()[-1][:-1])
    total_count = int(parts[1].split(',')[2].split()[-1])
    distances.append(distance)
    mean_errors.append(mean_error)
    std_devs.append(std_dev)
    total_counts.append(total_count)

# トータルカウントの合計値を計算
total_count_sum = sum(total_counts)

# ヒストグラムの作成
plt.figure(figsize=(12, 8))
bars = plt.bar([d + 0.25 for d in distances], mean_errors, yerr=std_devs, width=0.5, color='skyblue', edgecolor='black', alpha=0.7, capsize=5)
plt.xlabel('Distance (m)', fontsize=14)
plt.ylabel('Overall Mean Error (m)', fontsize=14)
plt.title('Overall Mean Error by Distance with Standard Deviation', fontsize=16)
plt.grid(True)

# 境界線の追加
for bar in bars:
    bar.set_edgecolor('black')

# 平均値をバーの上に表示（標準偏差の線のほんの少し上に表示）
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + std_devs[i] + 0.05, f'{mean_errors[i]:.2f}', ha='center', va='bottom', fontsize=12, color='black')

# すべての距離の目盛りを表示
bins = np.arange(0, 12, 0.5)
plt.xticks(bins, fontsize=12)
plt.yticks(fontsize=12)

# トータルカウントの合計値を右上に表示
plt.text(0.95, 0.95, f'Total Count: {total_count_sum}', ha='right', va='top', transform=plt.gca().transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

plt.show()