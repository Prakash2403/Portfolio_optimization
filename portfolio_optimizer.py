import pandas as pd
import matplotlib.pyplot as plt
import quandl
import scipy.optimize
import numpy as np
from config import *

quandl.ApiConfig.api_key = QUANDL_KEY

NEW_DF = None


def plot_df(df):
    df.plot()
    plt.show()


def get_daily_returns(data: pd.DataFrame, feature='port_vals'):
    """
    Daily return for kth day is given by

    r = data[k]/data[k-1] - 1

    :param data: Pandas dataframe containing stock price data.
    :param feature: feature on which daily return has to be calculated.

    :return: daily returns on given feature.
    """
    return data[feature] / data[feature].shift() - 1


def get_data_from_quandl(stocks, save=True):
    df_list = []
    for stock in stocks:
        df = quandl.get(stock, start_date=START_DATE, end_date=END_DATE)
        df_list.append(df)
        if save:
            df.to_csv('datasets/{0}.csv'.format(stock))
    return df_list


def optimize_portfolio(initial_portfolio):
    a = scipy.optimize.minimize(
        f, initial_portfolio,
        constraints={'type': 'eq', 'fun': constraint},
        bounds=((0, 1), (0, 1), (0, 1), (0, 1))
    )
    return a


def get_sharpe_ratio(portfolio_returns, k=np.math.sqrt(252)):
    """
    Assumption: Risk free rate is 0.

    Actual formula: k * mean(R_p - R_f) / std(R_p - R_f)
    R_p: Daily stock returns.
    R_f: Risk free rates
    k : sqrt(252), number of days stocks have been traded.
    :param k: sqrt(252), square root of number of days stocks have been traded.
    :param portfolio_returns: R_p: Daily stock returns.
    :param portfolio_returns: R_f: Risk free rates
    :return: sharpe ratio for given stock
    """
    return k * np.mean(portfolio_returns) / np.std(portfolio_returns)


def get_cumulative_return(portfolio_returns):
    return portfolio_returns.iloc[-1] / portfolio_returns.iloc[0]


def merge_dataframes(stocks, dataframes: list):
    new_dataframe = pd.DataFrame(index=pd.date_range(START_DATE, END_DATE))
    for i, dataframe in enumerate(dataframes):
        if type(dataframe.index) != pd.DatetimeIndex:
            dataframe.index = pd.to_datetime(dataframe['Date'])
        dataframe = dataframe.rename(columns={"Adj_Close": stocks[i].split('.')[0]})
        new_dataframe = new_dataframe.join(dataframe[[stocks[i].split('.')[0]]], how='inner')
    return new_dataframe


def f(x):
    global NEW_DF
    NEW_DF = NEW_DF/NEW_DF.iloc[0]
    x_new = x * NEW_DF * ASSET
    x_new['port_vals'] = x_new.sum(axis=1)
    daily_returns = get_daily_returns(x_new)
    return -1 * get_sharpe_ratio(daily_returns)


def constraint(x):
    return sum(x) - 1


def get_datasets(stocks, save=True, mode='LOCAL'):
    if mode == 'LOCAL':
        return [pd.read_csv(DATASET_PATH + stock + '.csv') for stock in stocks]
    if mode == 'QUANDL':
        return get_data_from_quandl(stocks, save)


def main():
    stocks = ['EOD/DIS', 'EOD/MSFT', 'EOD/BA', 'EOD/JNJ']
    dataframes = get_datasets(stocks=stocks)
    initial_portfolio = np.ones((1, 4))/4
    new_df = merge_dataframes(stocks, dataframes)
    global NEW_DF
    NEW_DF = new_df
    det = optimize_portfolio(initial_portfolio)
    final_portfolio = det['x']
    print(det)
    final_state = final_portfolio * NEW_DF * ASSET
    print("Total Asset")
    print(ASSET)
    print("Mean Final return.")
    print(round(final_state.sum(axis=1).mean(), 3))
    print("Final state of portfolio is decided by maximizing sharpe ratio.")


if __name__ == "__main__":
    main()
