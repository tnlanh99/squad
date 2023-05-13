import os
import numpy as np


def split_data(data, n_part):
    n = data.shape[0]
    n_part = min(n_part, n)
    n_per_part = n // n_part
    parts = []
    for i in range(n_part):
        if i == n_part - 1:
            parts.append(data[i * n_per_part:])
        else:
            parts.append(data[i * n_per_part:(i + 1) * n_per_part])
    return parts


if __name__ == "__main__":
    n_part = 5
    data = np.load("./data/train.npz")
    splited_data = [{} for _ in range(n_part)]

    for file in data.files:
        print(f"Processing {file} {data[file].shape}...")
        parts = split_data(data[file], n_part)
        for i in range(len(parts)):
            splited_data[i][file] = parts[i]

    for i in range(n_part):
        np.savez(f"./data/train_part_{i}.npz", **splited_data[i])
        
    print("Done!")
