from __future__ import print_function
from __future__ import division

import os
import math
import random

import numpy as np
import pandas as pd

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def calc_geom(scores, num_predictions):
    result = scores[0]
    for i in range(1, num_predictions):
        result *= scores[i]
    result = math.pow(result, 1.0 / num_predictions)
    return result

def calc_geom_arr(predictions_total, num_predictions):
    results = np.array(predictions_total[0])
    for i in range(1, num_predictions):
        results *= np.array(predictions_total[i])
    results = np.power(results, 1.0 / num_predictions)
    return results.tolist()

def write_submission(predictions, ids, dest):
    df = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', pd.Series(ids, index=df.index))
    df.to_csv(dest, index=False)
