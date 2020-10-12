import numpy as np


def generate_sequence_data(data, history_len):
    x, y = [], []
    for i in range(history_len, len(data)):
        x.append(data[i-history_len:i, 0])
        y.append(data[i, 0])
    x, y = np.array(x), np.array(y)
    return x, y
