def calculate_z_score(history, current):
    import numpy as np
    arr = [float(x) for x in history]
    if len(arr) < 3:
        return 0.0
    mu = sum(arr) / len(arr)
    variance = sum((x - mu)**2 for x in arr) / len(arr)
    std = variance**0.5
    if std < 1e-9:
        return 0.0
    return (float(current) - mu) / std
