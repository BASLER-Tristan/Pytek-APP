import random
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import string

list_database = [
    "TSF_Project\\database\\data_R.csv",
    "TSF_Project\\database\\data_WN.csv",
    "TSF_Project\\database\\data_WYI.csv",
    "TSF_Project\\database\\data_YI.csv",
]

data_YI = pd.read_csv('TSF_Project\\database\\data_YI.csv')
data_YI.set_index(pd.to_datetime(data_YI['Unnamed: 0']),inplace=True)
data_YI.drop(columns='Unnamed: 0',inplace=True)

data_WN = pd.read_csv('TSF_Project\\database\\data_WN.csv')
data_WN.set_index(pd.to_datetime(data_WN['Unnamed: 0']),inplace=True)
data_WN.drop(columns='Unnamed: 0',inplace=True)

data_WYI = pd.read_csv('TSF_Project\\database\\data_WYI.csv')
data_WYI.set_index(pd.to_datetime(data_WYI['Unnamed: 0']),inplace=True)
data_WYI.drop(columns='Unnamed: 0',inplace=True)

data_R = pd.read_csv('TSF_Project\\database\\data_R.csv')
data_R.set_index(pd.to_datetime(data_R['Unnamed: 0']),inplace=True)
data_R.drop(columns='Unnamed: 0',inplace=True)

database_explanation={}
database_explanation["data_YI"]=("The dataset translates the sales of ten different products, {} " \
                                                  "between january 2010 and december 2019. \n"
                                               "The monthly purchase has been created by a white noise for every"
                                                   "year with a mean value for the first year and inscrease by 120% each year"
                                                  "and a standard deviation of 4.".format(" ,".join(data_YI.columns)))
database_explanation["data_WYI"]=("The dataset translates the sales of ten different products, {} " \
                                                  "between january 2010 and december 2019. \n"
                                               "The monthly purchase has been created by a white noise for every"
                                                   "year with a mean value for the first year and inscrease by 120% each year"
                                                  "and a standard deviation of 4.".format(" ,".join(data_WYI.columns)))
database_explanation["data_WN"]=("The dataset translates the sales of ten different products, {} " \
                                                  "between january 2010 and december 2019. \n"
                                               "The monthly purchase has been created by a white noise for every"
                                                   "year with a mean value for the first year and inscrease by 120% each year"
                                                  "and a standard deviation of 4.".format(" ,".join(data_WN.columns)))
database_explanation["data_R"]=("The dataset translates the sales of ten different products, {} " \
                                                  "between january 2010 and december 2019. \n"
                                               "The monthly purchase has been created by a white noise for every"
                                                   "year with a mean value for the first year and inscrease by 120% each year"
                                                  "and a standard deviation of 4.".format(" ,".join(data_R.columns)))
data_columns={}
data_columns['data_R']=data_R.columns
data_columns['data_WYI']=data_WYI.columns
data_columns['data_WN']=data_WN.columns
data_columns['data_YI']=data_YI.columns

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

    data.to_csv("D:\\Python_Project\\reflect\\TSF_Project\\database\\data_YI.csv")

    ### Creation of the data without yearly increased
    N = 10
    K = 10
    data = []
    for i in range(N):
        mean = random.randint(10, 50)
        std = 4
        samples = np.random.normal(mean, std, size=12 * K)
        row = np.array([i for i in range(K*12)])
        row= row + samples
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

    data.to_csv("D:\\Python_Project\\reflect\\TSF_Project\\database\\data_WYI.csv")

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

    data.to_csv("D:\\Python_Project\\reflect\\TSF_Project\\database\\data_WN.csv")

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

    data.to_csv("D:\\Python_Project\\reflect\\TSF_Project\\database\\data_R.csv")

