##################################################################################
"""
This script contains the creation of the dataset
Running the script creates four dataset :
 - data_R, created by the numpy random function
 - data_WN, created by a white noise
 - data_WYI, created by a sequence between 0 and K modulated by a white noise
 - data_YI, created by yearly sequence increased by 20% each year and modulated by a white noise

N is the number of product
K is the number of month to generate. The first month is always January 2010.
"""
##################################################################################

import random
import numpy as np
import pandas as pd
import string
from os.path import join


database_explanation = {
    "data_YI": "a white noise for every year with a mean value for the first year and increase by 20% each year and a standard deviation of 4",
    "data_WYI": "a sequence from one to 120 modulated by a white noise with a standard deviation of 4",
    "data_WN": "a white noise with a standard deviation of 4",
    "data_R": "the numpy random function",
}

data_columns = {}
for name, description in database_explanation.items():
    data = pd.read_csv(join("TSF_Project", "database", f"{name}.csv"))
    data.set_index(pd.to_datetime(data["Unnamed: 0"]), inplace=True)
    data.drop(columns="Unnamed: 0", inplace=True)
    data_columns[name] = data.columns


def save_data(data, name):
    data.to_csv(join("TSF_Project", "database", f"data_{name}.csv"))


########################## Data Creation ##########################
if __name__ == "__main__":
    ### Creation of the yearly increased data
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

    save_data(data, "YI")

    ### Creation of the data without yearly increased
    N = 10
    K = 10
    data = []
    for i in range(N):
        mean = random.randint(10, 50)
        std = 4
        samples = np.random.normal(mean, std, size=12 * K)
        row = np.array([i for i in range(K * 12)])
        row = row + samples
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

    save_data(data, "WYI")

    ### Creation of the white noise data
    N = 10
    K = 10
    data = []
    for i in range(N):
        mean = random.randint(10, 50)
        std = 4
        samples = np.random.normal(mean, std, size=12 * K)
        data.append(samples)

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

    save_data(data, "WN")

    ### Creation of the white noise data
    N = 10
    K = 10
    data = []
    for i in range(N):
        samples = np.random.random(size=12 * K)
        data.append(samples)

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

    save_data(data, "R")
