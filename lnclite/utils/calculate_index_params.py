import math


def calculate_index_params(n_rows: int, dimension: int):
    # 1. 計算 IVF Partitions
    # 至少要有 256 筆數據才能訓練一個 partition
    max_partitions = n_rows // 256
    num_partitions = min(int(math.sqrt(n_rows) * 8), max_partitions)

    # 確保至少為 1 (雖然實務上 n 應該要更大)
    num_partitions = max(num_partitions, 1)

    # 2. 計算 PQ Sub-vectors
    # 尋找能整除維度且接近 D/8 的數
    target = dimension // 8
    # 簡單迴圈找出最大因數
    num_sub_vectors = 1
    for i in range(target, 0, -1):
        if dimension % i == 0:
            num_sub_vectors = i
            break

    return num_partitions, num_sub_vectors


# 假設 n = 1,000,000, D = 128
n = 1_000_000
d = 128
parts, subs = calculate_index_params(n, d)
print(f"建議參數: num_partitions={parts}, num_sub_vectors={subs}")
# 輸出參考: num_partitions=8000, num_sub_vectors=16
