import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# データセット1 yolo_n
data1 = """
Distance 1.0m - Overall Mean error: 4.678m, Overall Std: 0.000m, Total Count: 1
Distance 1.5m - Overall Mean error: 1.772m, Overall Std: 0.050m, Total Count: 7
Distance 2.0m - Overall Mean error: 0.374m, Overall Std: 0.301m, Total Count: 42
Distance 2.5m - Overall Mean error: 0.682m, Overall Std: 0.628m, Total Count: 92
Distance 3.0m - Overall Mean error: 0.381m, Overall Std: 0.113m, Total Count: 134
Distance 3.5m - Overall Mean error: 0.612m, Overall Std: 0.469m, Total Count: 156
Distance 4.0m - Overall Mean error: 0.418m, Overall Std: 0.150m, Total Count: 226
Distance 4.5m - Overall Mean error: 0.662m, Overall Std: 0.374m, Total Count: 251
Distance 5.0m - Overall Mean error: 0.646m, Overall Std: 0.352m, Total Count: 187
Distance 5.5m - Overall Mean error: 0.689m, Overall Std: 0.351m, Total Count: 178
Distance 6.0m - Overall Mean error: 0.759m, Overall Std: 0.262m, Total Count: 227
Distance 6.5m - Overall Mean error: 0.784m, Overall Std: 0.324m, Total Count: 251
Distance 7.0m - Overall Mean error: 0.755m, Overall Std: 0.357m, Total Count: 319
Distance 7.5m - Overall Mean error: 0.764m, Overall Std: 0.382m, Total Count: 354
Distance 8.0m - Overall Mean error: 0.898m, Overall Std: 0.429m, Total Count: 305
Distance 8.5m - Overall Mean error: 1.001m, Overall Std: 0.403m, Total Count: 315
Distance 9.0m - Overall Mean error: 1.056m, Overall Std: 0.362m, Total Count: 345
Distance 9.5m - Overall Mean error: 1.319m, Overall Std: 0.370m, Total Count: 362
Distance 10.0m - Overall Mean error: 1.897m, Overall Std: 0.441m, Total Count: 418
Distance 10.5m - Overall Mean error: 2.442m, Overall Std: 0.425m, Total Count: 308
Distance 11.0m - Overall Mean error: 1.867m, Overall Std: 0.444m, Total Count: 404
Distance 11.5m - Overall Mean error: 2.638m, Overall Std: 0.575m, Total Count: 285
"""

# データセット2 yolo_s
data2 = """
Distance 1.0m - Overall Mean error: 4.738m, Overall Std: 0.000m, Total Count: 1
Distance 1.5m - Overall Mean error: 0.626m, Overall Std: 0.104m, Total Count: 4
Distance 2.0m - Overall Mean error: 0.346m, Overall Std: 0.086m, Total Count: 33
Distance 2.5m - Overall Mean error: 0.576m, Overall Std: 0.434m, Total Count: 85
Distance 3.0m - Overall Mean error: 0.441m, Overall Std: 0.213m, Total Count: 127
Distance 3.5m - Overall Mean error: 0.419m, Overall Std: 0.117m, Total Count: 143
Distance 4.0m - Overall Mean error: 0.425m, Overall Std: 0.117m, Total Count: 216
Distance 4.5m - Overall Mean error: 0.681m, Overall Std: 0.368m, Total Count: 249
Distance 5.0m - Overall Mean error: 0.660m, Overall Std: 0.374m, Total Count: 180
Distance 5.5m - Overall Mean error: 0.729m, Overall Std: 0.312m, Total Count: 170
Distance 6.0m - Overall Mean error: 0.780m, Overall Std: 0.279m, Total Count: 221
Distance 6.5m - Overall Mean error: 0.710m, Overall Std: 0.280m, Total Count: 245
Distance 7.0m - Overall Mean error: 0.750m, Overall Std: 0.323m, Total Count: 283
Distance 7.5m - Overall Mean error: 0.769m, Overall Std: 0.383m, Total Count: 339
Distance 8.0m - Overall Mean error: 0.891m, Overall Std: 0.407m, Total Count: 270
Distance 8.5m - Overall Mean error: 1.052m, Overall Std: 0.413m, Total Count: 297
Distance 9.0m - Overall Mean error: 1.027m, Overall Std: 0.428m, Total Count: 321
Distance 9.5m - Overall Mean error: 1.154m, Overall Std: 0.395m, Total Count: 317
Distance 10.0m - Overall Mean error: 1.297m, Overall Std: 0.467m, Total Count: 396
Distance 10.5m - Overall Mean error: 1.836m, Overall Std: 0.612m, Total Count: 274
Distance 11.0m - Overall Mean error: 1.915m, Overall Std: 0.583m, Total Count: 316
Distance 11.5m - Overall Mean error: 2.185m, Overall Std: 0.618m, Total Count: 258
"""

# データセット3 yolo_m
data3 = """
Distance 1.0m - Overall Mean error: 4.777m, Overall Std: 0.000m, Total Count: 1
Distance 1.5m - Overall Mean error: 0.737m, Overall Std: 0.052m, Total Count: 4
Distance 2.0m - Overall Mean error: 0.319m, Overall Std: 0.078m, Total Count: 44
Distance 2.5m - Overall Mean error: 0.577m, Overall Std: 0.446m, Total Count: 91
Distance 3.0m - Overall Mean error: 0.376m, Overall Std: 0.112m, Total Count: 132
Distance 3.5m - Overall Mean error: 0.419m, Overall Std: 0.123m, Total Count: 145
Distance 4.0m - Overall Mean error: 0.415m, Overall Std: 0.137m, Total Count: 219
Distance 4.5m - Overall Mean error: 0.644m, Overall Std: 0.369m, Total Count: 257
Distance 5.0m - Overall Mean error: 0.649m, Overall Std: 0.384m, Total Count: 176
Distance 5.5m - Overall Mean error: 0.672m, Overall Std: 0.329m, Total Count: 169
Distance 6.0m - Overall Mean error: 0.782m, Overall Std: 0.226m, Total Count: 223
Distance 6.5m - Overall Mean error: 0.755m, Overall Std: 0.322m, Total Count: 229
Distance 7.0m - Overall Mean error: 0.719m, Overall Std: 0.309m, Total Count: 271
Distance 7.5m - Overall Mean error: 0.800m, Overall Std: 0.528m, Total Count: 346
Distance 8.0m - Overall Mean error: 0.869m, Overall Std: 0.399m, Total Count: 286
Distance 8.5m - Overall Mean error: 1.030m, Overall Std: 0.468m, Total Count: 296
Distance 9.0m - Overall Mean error: 1.048m, Overall Std: 0.413m, Total Count: 326
Distance 9.5m - Overall Mean error: 1.264m, Overall Std: 0.382m, Total Count: 316
Distance 10.0m - Overall Mean error: 1.319m, Overall Std: 0.462m, Total Count: 382
Distance 10.5m - Overall Mean error: 1.878m, Overall Std: 0.477m, Total Count: 283
Distance 11.0m - Overall Mean error: 2.386m, Overall Std: 0.395m, Total Count: 364
Distance 11.5m - Overall Mean error: 2.093m, Overall Std: 0.512m, Total Count: 310
"""

# データセット4 yolo_l
data4 = """
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

def parse_data(data):
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

    return distances, mean_errors, std_devs, total_counts

distances1, mean_errors1, std_devs1, total_counts1 = parse_data(data1)
distances2, mean_errors2, std_devs2, total_counts2 = parse_data(data2)
distances3, mean_errors3, std_devs3, total_counts3 = parse_data(data3)
distances4, mean_errors4, std_devs4, total_counts4 = parse_data(data4)

# コサイン類似度の計算
mean_errors1 = np.array(mean_errors1).reshape(1, -1)
mean_errors2 = np.array(mean_errors2).reshape(1, -1)
mean_errors3 = np.array(mean_errors3).reshape(1, -1)
mean_errors4 = np.array(mean_errors4).reshape(1, -1)

cosine_sim_12 = cosine_similarity(mean_errors1, mean_errors2)[0][0]
cosine_sim_13 = cosine_similarity(mean_errors1, mean_errors3)[0][0]
cosine_sim_14 = cosine_similarity(mean_errors1, mean_errors4)[0][0]
cosine_sim_23 = cosine_similarity(mean_errors2, mean_errors3)[0][0]
cosine_sim_24 = cosine_similarity(mean_errors2, mean_errors4)[0][0]
cosine_sim_34 = cosine_similarity(mean_errors3, mean_errors4)[0][0]

print("Cosine Similarity between Dataset 1 and 2:", cosine_sim_12)
print("Cosine Similarity between Dataset 1 and 3:", cosine_sim_13)
print("Cosine Similarity between Dataset 1 and 4:", cosine_sim_14)
print("Cosine Similarity between Dataset 2 and 3:", cosine_sim_23)
print("Cosine Similarity between Dataset 2 and 4:", cosine_sim_24)
print("Cosine Similarity between Dataset 3 and 4:", cosine_sim_34)