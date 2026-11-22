from collections import defaultdict
import numpy as np


def mean_metrics(rows):
    agg = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                agg[k].append(float(v))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
