# AR example
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

# data=pd.read_csv('reflect\\TSF_Project\\database\\data.csv')
# data.set_index(pd.to_datetime(data['Unnamed: 0']),inplace=True)
# data.drop(columns='Unnamed: 0',inplace=True)

support_type = [
    "AR",
    "MA",
    "ARMA",
    "ARIMA",
    "SARIMA",
    "SARIMAX",
    "VAR",
    "SES",
    "HWES",
]


class predictor:
    def __init__(self, type):
        if not (type is support_type):
            assert "Not support type for the moment"
        self.type = type

    def predict(self, data, date, N):
        if self.type == "AR":
            data, dff, dffm = AR_predict(data, date, N)
        elif self.type == "MA":
            data, dff, dffm = MA_predict(data, date, N)
        elif self.type == "ARMA":
            data, dff, dffm = ARMA_predict(data, date, N)
        elif self.type == "ARIMA":
            data, dff, dffm = ARIMA_predict(data, date, N)
        elif self.type == "SARIMA":
            data, dff, dffm = SARIMA_predict(data, date, N)
        elif self.type == "VAR":
            data, dff, dffm = VAR_predict(data, date, N)
        elif self.type == "SES":
            data, dff, dffm = SES_predict(data, date, N)
        elif self.type == "HWES":
            data, dff, dffm = HWES_predict(data, date, N)
        return data, dff, dffm


# AR
def AR_predict(data, date, N):
    cols = []
    col_name = []
    metrics = []
    col_name_metrics = []
    date_f = date + pd.DateOffset(months=N)
    for col in data.columns:
        row = data[col]
        train = row[row.index < date]
        test = row[row.index >= date]
        test = test[test.index < date_f]
        model = AutoReg(train, lags=N)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(start=len(train), end=len(train) + N - 1)
        cols.append(yhat)
        col_name.append(str(col) + "_AR")
        y = yhat[: len(test)]
        metrics.append(np.sqrt((np.array(test) - np.array(y)) ** 2))
        col_name_metrics.append(str(col) + "_error_AR")

    index = pd.date_range(date, periods=N, freq="MS")
    cols = np.transpose(cols)
    dff = pd.DataFrame(cols, columns=col_name, index=index)

    metrics = np.transpose(metrics)
    index = pd.date_range(date, periods=metrics.shape[0], freq="MS")
    dffm = pd.DataFrame(metrics, columns=col_name_metrics, index=index)
    data = data[data.index <= date_f]
    return data, dff, dffm


# MA
def MA_predict(data, date, N):
    cols = []
    col_name = []
    metrics = []
    col_name_metrics = []
    date_f = date + pd.DateOffset(months=N)
    for col in data.columns:
        row = data[col]
        train = row[row.index < date]
        test = row[row.index >= date]
        test = test[test.index < date_f]
        model = ARIMA(train, order=(0, 0, 1))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(
            start=len(train),
            end=len(train) + N - 1,
        )
        cols.append(yhat)
        col_name.append(str(col) + "_MA")
        y = yhat[: len(test)]
        metrics.append(np.sqrt((np.array(test) - np.array(y)) ** 2))
        col_name_metrics.append(str(col) + "_error_AR")

    index = pd.date_range(date, periods=N, freq="MS")
    cols = np.transpose(cols)
    dff = pd.DataFrame(cols, columns=col_name, index=index)

    metrics = np.transpose(metrics)
    index = pd.date_range(date, periods=metrics.shape[0], freq="MS")
    dffm = pd.DataFrame(metrics, columns=col_name_metrics, index=index)
    data = data[data.index < date_f]
    return data, dff, dffm


# ARMA
def ARMA_predict(data, date, N):
    cols = []
    col_name = []
    metrics = []
    col_name_metrics = []
    date_f = date + pd.DateOffset(months=N)
    for col in data.columns:
        row = data[col]
        train = row[row.index < date]
        test = row[row.index >= date]
        test = test[test.index < date_f]
        model = ARIMA(train, order=(2, 0, 1))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(
            start=len(train),
            end=len(train) + N - 1,
        )
        cols.append(yhat)
        col_name.append(str(col) + "_ARMA")
        y = yhat[: len(test)]
        metrics.append(np.sqrt((np.array(test) - np.array(y)) ** 2))
        col_name_metrics.append(str(col) + "_error_AR")

    index = pd.date_range(date, periods=N, freq="MS")
    cols = np.transpose(cols)
    dff = pd.DataFrame(cols, columns=col_name, index=index)

    metrics = np.transpose(metrics)
    index = pd.date_range(date, periods=metrics.shape[0], freq="MS")
    dffm = pd.DataFrame(metrics, columns=col_name_metrics, index=index)
    data = data[data.index < date_f]
    return data, dff, dffm


# ARIMA
def ARIMA_predict(data, date, N):
    cols = []
    col_name = []
    metrics = []
    col_name_metrics = []
    date_f = date + pd.DateOffset(months=N)
    for col in data.columns:
        row = data[col]
        train = row[row.index < date]
        test = row[row.index >= date]
        test = test[test.index < date_f]
        model = ARIMA(train, order=(1, 1, 1))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(
            start=len(train),
            end=len(train) + N - 1,
        )
        cols.append(yhat)
        col_name.append(str(col) + "_ARiMA")
        y = yhat[: len(test)]
        metrics.append(np.sqrt((np.array(test) - np.array(y)) ** 2))
        col_name_metrics.append(str(col) + "_error_ARIMA")

    index = pd.date_range(date, periods=N, freq="MS")
    cols = np.transpose(cols)
    dff = pd.DataFrame(cols, columns=col_name, index=index)

    metrics = np.transpose(metrics)
    index = pd.date_range(date, periods=metrics.shape[0], freq="MS")
    dffm = pd.DataFrame(metrics, columns=col_name_metrics, index=index)
    data = data[data.index < date_f]
    return data, dff, dffm


# SARIMA
def SARIMA_predict(data, date, N):
    cols = []
    col_name = []
    metrics = []
    col_name_metrics = []
    date_f = date + pd.DateOffset(months=N)
    for col in data.columns:
        row = data[col]
        train = row[row.index < date]
        test = row[row.index >= date]
        test = test[test.index < date_f]
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(
            start=len(train),
            end=len(train) + N - 1,
        )
        cols.append(yhat)
        col_name.append(str(col) + "_SARIMA")
        y = yhat[: len(test)]
        metrics.append(np.sqrt((np.array(test) - np.array(y)) ** 2))
        col_name_metrics.append(str(col) + "_error_SARIMA")

    index = pd.date_range(date, periods=N, freq="MS")
    cols = np.transpose(cols)
    dff = pd.DataFrame(cols, columns=col_name, index=index)

    metrics = np.transpose(metrics)
    index = pd.date_range(date, periods=metrics.shape[0], freq="MS")
    dffm = pd.DataFrame(metrics, columns=col_name_metrics, index=index)
    data = data[data.index < date_f]
    return data, dff, dffm


# VAR
def VAR_predict(data, date, N):
    data_train = data[data.index < date]
    model = VAR(data_train)
    date_f = date + pd.DateOffset(months=N)
    data_test = data[data.index >= date]
    data_test = data_test[data_test.index < date_f]
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.forecast(np.array(data_train), steps=N)
    index = pd.date_range(date, periods=N, freq="MS")
    col_name = []
    for col in data.columns:
        col_name.append(str(col) + "_VAR")
    dff = pd.DataFrame(yhat, columns=col_name, index=index)
    dffm = (np.array(data_test) - np.array(y)) ** 2
    dffm = pd.DataFrame(dffm)
    dffm.index = data_test.index
    dffm.columns = [col + "_error_VAR" for col in data_test.columns]
    data = data[data.index < date_f]
    return data, dff, dffm


# SES
def SES_predict(data, date, N):
    cols = []
    col_name = []
    metrics = []
    col_name_metrics = []
    date_f = date + pd.DateOffset(months=N)
    for col in data.columns:
        row = data[col]
        train = row[row.index < date]
        test = row[row.index >= date]
        test = test[test.index < date_f]
        col_values = []
        for i in range(N):
            model = SimpleExpSmoothing(train)
            model_fit = model.fit()
            yhat = model_fit.predict(start=len(train), end=len(train))
            train = train.append(yhat)
            col_values.append(float(yhat))
        cols.append(yhat)
        col_name.append(str(col) + "_AR")
        y = yhat[: len(test)]
        metrics.append(np.sqrt((np.array(test) - np.array(y)) ** 2))
        col_name_metrics.append(str(col) + "_error_AR")

    index = pd.date_range(date, periods=N, freq="MS")
    cols = np.transpose(cols)
    dff = pd.DataFrame(cols, columns=col_name, index=index)

    metrics = np.transpose(metrics)
    index = pd.date_range(date, periods=metrics.shape[0], freq="MS")
    dffm = pd.DataFrame(metrics, columns=col_name_metrics, index=index)
    data = data[data.index < date_f]
    return data, dff, dffm


# HWES
def HWES_predict(data, date, N):
    cols = []
    col_name = []
    metrics = []
    col_name_metrics = []
    date_f = date + pd.DateOffset(months=N)
    for col in data.columns:
        row = data[col]
        train = row[row.index < date]
        test = row[row.index >= date]
        test = test[test.index < date_f]
        # make prediction
        col_values = []
        for i in range(N):
            model = ExponentialSmoothing(train)
            model_fit = model.fit()
            yhat = model_fit.predict(start=len(train), end=len(train))
            train = train.append(yhat)
            col_values.append(float(yhat))
        cols.append(yhat)
        col_name.append(str(col) + "_HWES")
        y = yhat[: len(test)]
        metrics.append(np.sqrt((np.array(test) - np.array(y)) ** 2))
        col_name_metrics.append(str(col) + "_error_HWES")

    index = pd.date_range(date, periods=N, freq="MS")
    cols = np.transpose(cols)
    dff = pd.DataFrame(cols, columns=col_name, index=index)

    metrics = np.transpose(metrics)
    index = pd.date_range(date, periods=metrics.shape[0], freq="MS")
    dffm = pd.DataFrame(metrics, columns=col_name_metrics, index=index)
    data = data[data.index < date_f]
    return data, dff, dffm
