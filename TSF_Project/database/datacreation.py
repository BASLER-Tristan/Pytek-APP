########################## Data Creation ##########################

import random
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import string
from pathlib import Path


if __name__ == "__main__":
    N = 10
    K = 10
    data = []
    for i in range(N):
        mean = random.randint(10, 50)
        std = 4
        row = []
        for j in range(K):
            samples = np.random.normal(mean, std, size=12)
            row = row + list(samples)
            mean = mean * 1.2
        data.append(row)

    cols = np.transpose(data)
    data = pd.DataFrame(cols)
    start = "2010-01-01"
    end = "2019-12-31"

    data.index = pd.date_range(start, end, freq="MS")

    list_product = []
    while len(list_product) < N:
        x = "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
        )
        if not (x in list_product):
            list_product.append(x)
    data.columns = list_product

    data.to_csv(Path("database/data.csv"))
