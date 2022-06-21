##################################################################################
"""



"""
##################################################################################

import pandas as pd
import numpy as np


def pipeline(data, size_rolling_windows=12):
    linearized_xdata = []
    linearized_ydata = []
    N = len(data)
    list_index = list(data.index)
    for i in range(size_rolling_windows, N):
        for col in data.columns:
            linearized_xdata.append(
                list(data.loc[list_index[(i - size_rolling_windows): i], col])
            )
    for i in range(size_rolling_windows, N - 1):
        for col in data.columns:
            linearized_ydata.append(data.loc[list_index[i + 1], col])
    linearized_ydata = linearized_ydata + [np.nan for i in range(len(data.columns))]
    linearized_date = list_index[size_rolling_windows:]
    linearized_date = [
        item for item in linearized_date for _ in range(len(data.columns))
    ]

    df_date = pd.DataFrame(linearized_date, columns=["date"])
    df_date["year"] = df_date.date.apply(lambda x: x.year)
    df_date["month"] = df_date.date.apply(lambda x: x.month)
    df_x = pd.DataFrame(linearized_xdata)
    df_y = pd.DataFrame(linearized_ydata)
    res = pd.concat([df_date, df_x, df_y], axis=1)
    res.columns = [
        "date",
        "year",
        "month",
        "M -12",
        "M -11",
        "M -10",
        "M -9",
        "M -8",
        "M -7",
        "M -6",
        "M -5",
        "M -4",
        "M -3",
        "M -2",
        "M -1",
        "M",
    ]
    return res


def fit_model(model, res, date_limit):
    X = res[res.date < date_limit][
        [
            "year",
            "month",
            "M -12",
            "M -11",
            "M -10",
            "M -9",
            "M -8",
            "M -7",
            "M -6",
            "M -5",
            "M -4",
            "M -3",
            "M -2",
            "M -1",
        ]
    ]
    y = res[res.date < date_limit]["M"]
    model = model.fit(X, y)
    return model


def result_pipeline(model, res, data, date, nb_prediction, size_rolling_windows=12):
    ### Overlap data
    y_pred = model.predict(
        res[
            [
                "year",
                "month",
                "M -12",
                "M -11",
                "M -10",
                "M -9",
                "M -8",
                "M -7",
                "M -6",
                "M -5",
                "M -4",
                "M -3",
                "M -2",
                "M -1",
            ]
        ]
    )
    date_scale = pd.to_datetime(res["date"]) + pd.DateOffset(months=1)
    products = list(data.columns) * int(len(y_pred) / len(list(data.columns)))
    result_on_training = pd.DataFrame([y_pred, date_scale, products]).transpose()
    past_prediction = pd.DataFrame()
    for product in list(data.columns):
        past_prediction[product] = result_on_training[
            result_on_training[2].str.startswith(product)
        ].reset_index()[0]
    past_prediction.index = date_scale.unique()

    ### metrics
    data_past = data[data.index <= date]
    dffm = (data_past - past_prediction) ** 2

    ### future data
    list_index = list(data_past.index)
    input = data_past.loc[list_index[-12:]]
    init_date = date
    init_month = init_date.month
    init_year = init_date.year
    for i in range(nb_prediction):
        dfx = data_past.loc[list_index[-12:]].transpose()
        dfx.columns = [
            "M -12",
            "M -11",
            "M -10",
            "M -9",
            "M -8",
            "M -7",
            "M -6",
            "M -5",
            "M -4",
            "M -3",
            "M -2",
            "M -1",
        ]
        dfx["year"] = init_year
        dfx["month"] = init_month

        init_date = init_date + pd.DateOffset(months=1)
        init_month = init_date.month
        init_year = init_date.year

        y_pred = model.predict(dfx)
        y_pred = pd.DataFrame(y_pred).transpose()
        y_pred.columns = data_past.columns
        y_pred.index = [init_date]
        data_past = pd.concat([data_past, y_pred], axis=0)

    dff = pd.concat(
        [past_prediction, data_past[data_past.index > past_prediction.index.max()]]
    )
    dff.columns = [col + "_xgboost" for col in data.columns]
    dffm.columns = [col + "_error_xgboost" for col in data.columns]

    return data, dff, dffm
