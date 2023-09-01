# Damir Pavlin, Moscow 2021; damirxpavlin@gmail.com
import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt
import scipy.optimize as optimize
from datetime import datetime as dt
from datetime import timedelta as td

# mean-variance analysis
# it's day-to-day model, also let's try month-to-month and year-to-year and compare results
# TODO visualisation tool
# TODO: set restrictions
#  Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP
#  (or up to a total of 48,000 requests a day).
class Portfolio:
    def __init__(self, shares, quantity, start_prices, start_dates):
        self.shares = shares            # list with share names
        self.share_data = yf.download(  # download Df with historical data
                                        tickers=" ".join(shares),
                                        period="max",
                                        interval="1d",
                                        group_by='ticker',
                                        auto_adjust=True,
                                        prepost=True,
                                        threads=True,
                                        proxy=None)
        self.return_df = pd.DataFrame()
        self.stdevs = {}
        self.current_values = {}

        k = 0
        for share in self.shares:  # adding log return
            if len(self.shares) == 1:
                self.share_data['Log_ret'] = np.log(self.share_data['Close']) - \
                                                      np.log(self.share_data['Close'].shift(1))
                self.return_df[share] = self.share_data['Log_ret']
                self.stdevs[share] = np.std(self.share_data['Log_ret'])
                self.current_values[share] = self.share_data['Close'][-1] * quantity[k]
            else:
                self.share_data[(share, 'Log_ret')] = np.log(self.share_data[(share, 'Close')]) -\
                                                  np.log(self.share_data[(share, 'Close')].shift(1))
                self.return_df[share] = self.share_data[(share, 'Log_ret')]
                self.stdevs[share] = np.std(self.share_data[(share, 'Log_ret')])
                self.current_values[share] = self.share_data[(share, 'Close')][-1] * quantity[k]
            k += 1

        self.quantity = quantity    # list with share quantity
        self.start_prices = start_prices    # list with share buy prices
        self.start_dates = start_dates  # list with share start dates
        self.total_value = sum(self.current_values.values())  # total value of portfolio
        self.portfolio_summ = self.total_value

        self.weights = {}
        self.exp_ret = {}  # expected return per share
        self.portfolio_exp_ret = 0  # expected return of portfolio
        for share in self.shares:
            self.weights[share] = self.current_values[share] / self.total_value  # calculate weights
            if len(self.shares) == 1:
                self.exp_ret[share] = self.share_data['Log_ret'].mean()
            else:
                self.exp_ret[share] = self.share_data[(share, 'Log_ret')].mean()
            self.portfolio_exp_ret += self.weights[share] * self.exp_ret[share]

        self.corr_coef = self.return_df.corr()  # correlation matrix

        self.general_variance = 0
        for share_i in self.shares:
            for share_j in self.shares:
                self.general_variance += self.weights[share_i] * self.weights[share_j] * \
                                         self.stdevs[share_i] * self.stdevs[share_j] * \
                                         self.corr_coef[share_i][share_j]


    def portfolio_return(self, quantity):
        portfolio_ret_calc = 0  # expected return of portfolio
        weights_calc = {}
        current_values_calc = {}
        k = 0
        for share in self.shares:
            current_values_calc[share] = self.share_data[(share, 'Close')][-1] * quantity[k]
            k += 1

        total_val_calc = sum(current_values_calc.values())
        k = 0
        for share in self.shares:
            weights_calc[share] = current_values_calc[share] / total_val_calc   # calculate weights
            portfolio_ret_calc += weights_calc[share] * self.exp_ret[share]
            k += 1
        general_variance_calc = 0
        for share_i in self.shares:
            for share_j in self.shares:
                general_variance_calc += weights_calc[share_i] * weights_calc[share_j] * \
                                         self.stdevs[share_i] * self.stdevs[share_j] * \
                                         self.corr_coef[share_i][share_j]
        return general_variance_calc / portfolio_ret_calc, general_variance_calc, portfolio_ret_calc

    def plot_bullet(self, ret_list, var_list, additional_dot):
        var_list = [x*253*100 for x in var_list]
        ret_list = [x * 253 * 100 for x in ret_list]
        plt.figure('Variance/ Return')
        plt.scatter(var_list, ret_list)

        if additional_dot:
            plt.scatter(additional_dot[0]*253*100, additional_dot[1]*253*100, color='red')
        return plt

    def randomize_portfolio(self, n, portf_sum, max_share_count, max_iterates):
        """
        Create list with random generated portfolios with selected papers and total summ
        :param n: count of portfolios
        :param portf_sum: maximum sum of papers in portfolio
        :param max_share_count: maximum count of per share
        :param max_iterates: maximum count of random iterates
        :return: list of dicts presenting portfolios
        """
        quantity_list = []
        for i in range(n):
            print(f'Epoche is {i}')
            flag = False
            share_quantity = {}
            current_values = {}
            total_value = 0
            iterates = 0
            while not flag:
                iterates += 1
                for share in self.shares:
                    share_quantity[share] = np.random.randint(1, max_share_count)
                    current_values[share] = self.share_data[(share, 'Close')][-1] * share_quantity[share]
                total_value = sum(current_values.values())
                if (total_value <= portf_sum and current_values not in quantity_list) or iterates > max_iterates:
                    flag = True
            if iterates <= max_iterates:
                quantity_list.append(share_quantity)
        return quantity_list

    def plot_randoms(self, quantity_list, additional_portfolio):
        """
        Plot results of random portfolio collecting
        :param quantity_list:
        :return:
        """
        rel_list = []
        var_list = []
        ret_list = []
        risk_ret_df = pd.DataFrame(columns=['rel', 'shares'])
        for share in quantity_list:
            another_tuple = self.portfolio_return(list(share.values()))
            risk_ret_df = risk_ret_df.append({'rel': another_tuple[0], 'shares': str(share)}, ignore_index=True)
            rel_list.append(another_tuple[0])
            var_list.append(another_tuple[1])
            ret_list.append(another_tuple[2])

        if additional_portfolio:
            another_tuple = self.portfolio_return(list(additional_portfolio.values()))
            bullet_plot = self.plot_bullet(ret_list, var_list, [another_tuple[1], another_tuple[2]])
        else:
            bullet_plot = self.plot_bullet(ret_list, var_list, False)

        rel_df = pd.DataFrame(columns=['rel_list'])
        rel_df['rel_list'] = rel_list
        rel_df = rel_df.sort_values(by='rel_list', ascending=True)
        rel_df = rel_df.reset_index(drop=True)
        rel_df = rel_df.reset_index()
        plt.figure('Risk/Profit')
        rel_plot = plt.scatter(rel_df['index'], rel_df['rel_list'])
        risk_ret_df = risk_ret_df.sort_values(by='rel', ascending=True)
        return bullet_plot, rel_plot, risk_ret_df[:1]

    def set_profit(self, profit):
        self.profit = profit

    def for_minimize(self, quantity):
        fun_profit = self.portfolio_return(list(quantity))
        if fun_profit[2] <= self.profit:
            return fun_profit[1]
        else:
            return np.inf

    def minimize_risk(self, initial_guess):
        result = optimize.minimize(self.for_minimize, initial_guess, method='Nelder-Mead')
        if result.success:
            result_list = list(result.x)
            result_list = [np.round(x, 0) for x in result_list]
            return result_list
        else:
            return result.message

    def add_deposit(self, percent, price):
        self.shares.append('Depo')
        self.share_data[('Depo', 'Close')] = ''
        first_year = self.share_data.index[0].year
        first_value = price
        for index, row in self.share_data.iterrows():
            if index.year == first_year:
                self.share_data.loc[index, ('Depo', 'Close')] = first_value
            else:
                first_value = first_value * percent
                first_year = index.year
                self.share_data.loc[index, ('Depo', 'Close')] = first_value

        self.share_data[('Depo', 'Close')] = self.share_data[('Depo', 'Close')].astype(float)
        self.share_data[('Depo', 'Log_ret')] = np.log(self.share_data[('Depo', 'Close')]) - \
                                               np.log(self.share_data[('Depo', 'Close')].shift(1))
        self.return_df['Depo'] = self.share_data[('Depo', 'Log_ret')]
        self.stdevs['Depo'] = np.std(self.share_data[('Depo', 'Log_ret')])
        self.current_values['Depo'] = self.share_data[('Depo', 'Close')][-1]
        self.total_value += price
        self.weights['Depo'] = self.current_values['Depo'] / self.total_value  # calculate weights
        self.exp_ret['Depo'] = self.share_data[('Depo', 'Log_ret')].mean()
        self.corr_coef = self.return_df.corr()


# myPortfolio = Portfolio(['AMD', 'KO', 'SAVE'], [1, 1, 1], [1, 1, 1], [1, 1, 1])
# myPortfolio = Portfolio(['AAPL', 'BAYN.DE', 'BTI', 'CAT', 'IBM', 'INTC', 'JD', 'NOV', 'NEM', 'OGZD.IL', 'HOOD', 'SWI',
#                          'VALE'],
#                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# quant_list = myPortfolio.randomize_portfolio(100, 1000, 10, 1000)
# myPortfolio.plot_randoms(quant_list)

# print(myPortfolio.stdevs)
# print(myPortfolio.corr_coef)
#
# plt.figure()
# sr = myPortfolio.share_data[('AMD', 'Close')]
# sr.plot(title='AMD prices')
# plt.show()


#  TODO
# Portfolio optimisation
# Recommendational system: too high correlation, higher\ lower than MA and so on

def correlation_calc(shares):
    """
    Get a correlation matrix from list of tickers
    :param shares: list with tickers
    :return: correlation matrix
    """
    share_data = yf.download(  # download Df with historical data
                                        tickers=" ".join(shares),
                                        period="max",
                                        interval="1d",
                                        group_by='ticker',
                                        auto_adjust=True,
                                        prepost=True,
                                        threads=True,
                                        proxy=None)
    return_df = pd.DataFrame()
    for share in shares:  # adding log return
        share_data[(share, 'Log_ret')] = np.log(share_data[(share, 'Close')]) - \
                                         np.log(share_data[(share, 'Close')].shift(1))
        return_df[share] = share_data[(share, 'Log_ret')]
    corr_coef_log_ret = return_df.corr()
    return corr_coef_log_ret


# TODO
# need always to improve this search function
def ticker_searcher(company_name):
    """
    Search stock ticker by its company's name
    :param company_name: str
    :return: found ticket: str / Not Found / dataframe with matches
    """
    ticker_base = pd.read_csv('secwiki_tickers.csv')
    return_df = ticker_base.where(ticker_base['Name'].str.contains(company_name, case=False)).dropna()
    if len(return_df) == 1:
        return return_df['Ticker'].to_list()[0]
    elif len(return_df) == 0:
        return 'Not Found'
    else:
        return return_df[['Ticker', 'Name', 'Sector', 'Industry']]


def check_ticker(ticker):
    """
    Check if ticker exist at SPB Exchange
    :param ticker:
    :return:
    """
    securities_df = pd.read_csv('ListingSecurityList.csv', sep=';', engine='python')
    ticker_base = pd.read_csv('secwiki_tickers.csv')
    securities_list = securities_df['s_RTS_code'].to_list() + ticker_base['Ticker'].to_list()
    return ticker in securities_list


def fast_corrs(tickers):
    """
    Function to count long and short correlations between instruments
    :param tickers: list with tickers
    :return:
    """

    # Long Correlation
    share_data = yf.download(  # download Df with historical data
        tickers=" ".join(tickers),
        period="max",
        interval="1d",
        group_by='ticker',
        auto_adjust=True,
        prepost=True,
        threads=True,
        proxy=None)
    close_data = share_data.xs('Close', level=1, axis=1)
    corr_matrix = close_data.dropna().corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # return
    sorted_corr_matrix = corr_matrix.abs().unstack().sort_values(kind="quicksort").reset_index()
    sorted_corr_matrix = sorted_corr_matrix[sorted_corr_matrix['level_0'] != sorted_corr_matrix['level_1']]  # return

    # Short Correlation (week)
    min_data = yf.download(
        tickers=" ".join(tickers),
        period="7d",
        interval="1m",
        group_by='ticker',
        auto_adjust=True,
        prepost=True,
        threads=True,
        proxy=None
    )
    previous_day = (dt.now() - td(days=1)).date()
    min_close_data = min_data.xs('Close', level=1, axis=1)
    min_close_data = min_close_data.fillna(method='ffill')

    # Calculate the correlation matrix for the weekly data
    weekly_corr_matrix = min_close_data.dropna().corr()

    previous_day = (dt.now() - td(days=1)).date()

    # Convert the index to date
    min_close_data.index = min_close_data.index.date

    # Select rows with the previous day
    previous_day_data = min_close_data[min_close_data.index == previous_day]

    # Calculate the correlation matrix for the last day
    daily_corr_matrix = previous_day_data.corr()  # return

    # Calculate the difference between the all-time correlation and the weekly and daily correlations
    weekly_diff_corr_matrix = corr_matrix - weekly_corr_matrix  # return
    daily_diff_corr_matrix = corr_matrix - daily_corr_matrix  # return

    # Create a mask to hide the upper triangle of the correlation matrix (since it's mirrored around its main diagonal)
    mask1 = np.triu(np.ones_like(corr_matrix, dtype=bool))  # return





