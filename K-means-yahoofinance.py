# coding=utf-8

import requests
import re
import json
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import xlwt

def retrieve_quotes_historical(stock_code):
    quotes = []
    url = 'https://finance.yahoo.com/quote/%s/history?p=%s' % (stock_code, stock_code)
    r = requests.get(url)
    m = re.findall('"HistoricalPriceStore":{"prices":(.*),"isPending"', r.text)
    if m:
        quotes = json.loads(m[0])
        quotes = quotes[::-1]
    result = []
    for it in quotes:
        if not 'type' in it:
            result.append(it)
    return result
    # return [item for item in quotes if not 'type' in item]


def create_df(stock_code):
    quotes = retrieve_quotes_historical(stock_code)
    list1 = ['close', 'date', 'high', 'low', 'open', 'volume']
    df_totalvolume = pd.DataFrame(quotes, columns=list1)
    return df_totalvolume


if __name__ == "__main__":
    listDji = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DD']
    listTemp = [0] * len(listDji)
    df_list = []
    for i in range(len(listTemp)):
        df = create_df(listDji[i])
        listTemp[i] = df.close
        df_list.append(df)
    status = [0]*len(listDji)
    for i in range(len(status)):
        status[i] = np.sign(np.diff(listTemp[i]))
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(status)
    pred = kmeans.predict(status)
    print(pred)
    df_list[0].to_csv("MMM_stock.csv")
    # dateFrame.read_csv()

    print(df_list[0])

