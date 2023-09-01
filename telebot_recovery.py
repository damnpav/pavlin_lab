import telebot
import sqlite3
from telebot import types
from datetime import datetime as dt
import matplotlib.pyplot as plt
import dataframe_image as dfi
import psycopg2
import os
import traceback

import models
from models import correlation_calc as cc

#token_path = r"/usr/local/projects/pavlin_lab/bot_token.txt"
token_path = 'bot_token.txt'
bot_token = open(token_path).readlines()[0].replace('\n', '')
home_path = r'/usr/projects/pavlin_lab/'
#home_path = r''
STAND_TYPE = 'TEST'  # PROD\TEST, influence on work with db

message_dict = {'welcome': 'Welcome to PavlinLab! \nThis bot provides different techniques to analyze and model '
                           'securities portfolios \nChoose your option:',
                'about': 'PavlinLab Project is a tool for modelling portfolio of securities with Modern Portfolio '
                         'Theory.',
                'contact': 'You may always contact us on e-mail: hcanilvap@gmail.com',
                'search': 'You may search for ticker of your company with /search command.\n\n '
                          'For example: \n/search Xerox',
                'correlation': 'To count correlation between some shares please type command \n/correlation and write '
                               'tickers of these papers, sep by comma.\n\nFor example: \n/correlation INTC, CSCO, '
                               'FB\n\nYou may search for ticker of your company with /search command.',
                'model': 'For creating model of your portfolio please type command \n/model and write tickers of '
                         'shares that you want to include in your portfolio, after "@" place count of each ticker '
                         'in the same order, all sep by comma.\n\nFor example:\n/model AMD, AMZN, MSFT, BK, '
                         'AAPL@1, 1, 2, 1, 4\n\nYou may search for ticker of your company with /search command.'}
top_users = ['dampall']


def replace_quotes(your_str):
    return your_str.replace('"', '').replace("'", "")


try:
    bot = telebot.TeleBot(bot_token)


    @bot.message_handler(commands=['start'])
    def start_handler(message):
        log_message(message, 'incoming_msg')
        chat_id = message.chat.id
        bot.send_message(chat_id, message_dict['welcome'], reply_markup=welcoming_buttons(), parse_mode='HTML')


    @bot.message_handler(commands=['model'])
    def model_handler(message):
        log_message(message, 'model_request')
        chat_id = message.chat.id
        share_info = message.text.replace('/model', '').replace(' ', '').split('@')
        share_list = share_info[0].split(',')
        count_list = share_info[1].split(',')
        count_list = list(map(int, count_list))  # from str to int
        bot.send_message(chat_id, 'Preparing calculations...', parse_mode='HTML')
        return_dict = print_out_model(share_list, count_list, message.from_user.username)
        if type(return_dict) != str:
            bot.send_message(chat_id, return_dict['return_str'], parse_mode='HTML')
            bot.send_photo(chat_id=chat_id, photo=open(return_dict['pie_path'], 'rb'),
                           caption='Weights of shares in portfolio', parse_mode='HTML')
            bot.send_photo(chat_id=chat_id, photo=open(return_dict['corr_path'], 'rb'),
                           caption='Correlations between papers', parse_mode='HTML')
            bot.send_message(chat_id, f'Total summ: {return_dict["total_summ"]}', parse_mode='HTML')
        else:
            bot.send_message(chat_id, return_dict, parse_mode='HTML')


    @bot.message_handler(commands=['correlation'])
    def correlation_handler(message):
        log_message(message, 'correlation_request')
        chat_id = message.chat.id
        shares_str = message.text.replace('/correlation', '')
        bot.send_message(chat_id, 'Preparing calculations...', parse_mode='HTML')
        return_str = cor_calc(shares_str, message.from_user.username)
        if 'PNGs' in return_str:
            bot.send_photo(chat_id=chat_id, photo=open(return_str, 'rb'), caption='Correlations between papers',
                           parse_mode='HTML')
        else:
            bot.send_message(chat_id, return_str, parse_mode='HTML')


    @bot.message_handler(commands=['search'])
    def search_handler(message):
        log_message(message, 'search_request')
        chat_id = message.chat.id
        companies_str = message.text.replace('/search', '').replace(' ', '')
        return_str = models.ticker_searcher(companies_str)
        if type(return_str) == str:
            bot.send_message(chat_id, return_str, parse_mode='HTML')
        else:
            df_styled = return_str[:10].style.background_gradient()
            found_path = f'{home_path}PNGs/found_companies_{dt.now().strftime("%H%M%S%d%m%Y")}.png'
            dfi.export(df_styled, found_path, table_conversion='matplotlib')
            bot.send_photo(chat_id=chat_id, photo=open(found_path, 'rb'), caption='Found companies',
                           parse_mode='HTML')


    @bot.callback_query_handler(func=lambda call: True)
    def handle_buttons(call):
        msg = str(call.data)
        log_message(call.message, msg)
        chat_id = call.message.chat.id
        if msg == '/about':
            bot.send_message(chat_id, message_dict['about'], reply_markup=welcoming_buttons(), parse_mode='HTML')
        elif msg == '/contact':
            bot.send_message(chat_id, message_dict['contact'], reply_markup=welcoming_buttons(), parse_mode='HTML')
        elif msg == '/searcher_cb':
            bot.send_message(chat_id, message_dict['search'], reply_markup=welcoming_buttons(), parse_mode='HTML')
        elif msg == '/corr_calc_cb':
            bot.send_message(chat_id, message_dict['correlation'], reply_markup=welcoming_buttons(), parse_mode='HTML')
        else:
            bot.send_message(chat_id, message_dict['model'], reply_markup=welcoming_buttons(), parse_mode='HTML')


    def print_out_model(shares, quantity, username):
        """
        Function to process portfolio construction with model's functions
        :param shares: List of shares
        :param quantity: List with counts of shares
        :param username: Username
        :return: dict with results
        """
        if len(shares) == 0:
            return f'Tickers are not found in your request. Please specify them like that: \n' \
                   f'/model AMZN, TWTR, CSCO'
        if len(shares) > 5 and username not in top_users:
            return f'No more than 5 papers! For more services contact us'
        for share in shares:
            if not models.check_ticker(share) and username not in top_users:
                return f'{share} ticker not found at our base.' \
                       f'You may search for ticker of your company with /search command.'
        portf = models.Portfolio(shares, quantity, [1] * len(shares), [1] * len(shares))
        return_str = f'Expectation return: {round(portf.portfolio_exp_ret * 253 * 100, 2)} % annually.\n' \
                     f'Risk: {round(portf.general_variance * 253 * 100, 2)} % annually.\n' \
                     f'Risk \ Reward ratio: {round(portf.general_variance / portf.portfolio_exp_ret, 2)}'
        pie_path = f'{home_path}PNGs/pie_chart_{dt.now().strftime("%H%M%S%d%m%Y")}.png'
        weights_pie(portf.weights, pie_path)  # make a pie chart with share's weights in portfolio
        df_styled = portf.corr_coef.style.background_gradient()
        corr_path = f'{home_path}PNGs/corr_df_{dt.now().strftime("%H%M%S%d%m%Y")}.png'
        dfi.export(df_styled, corr_path, table_conversion='matplotlib')
        return_dict = {'return_str': return_str, 'pie_path': pie_path, 'corr_path': corr_path,
                       'total_summ': portf.portfolio_summ}
        return return_dict


    def weights_pie(weights, pie_name):
        """
        Function to plot pies of portfolios
        :param weights: dict: keys - labels, values - weights
        :param pie_name: where to save picture of pie
        :return: None
        """
        labels = list(weights.keys())
        sizes = list(weights.values())
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig(pie_name)


    def cor_calc(tickers_str, username):
        """
        Function to calculate correlations between tickers with using model
        :param tickers_str: string with tickers
        :return: path to picture with correlation table
        """
        tickers_list = tickers_str.replace(' ', '').split(',')
        if len(tickers_list) < 2:
            return "Incorrect request. String should be in format like: \n/correlation AMZN, MSFT. \nIf you don't know " \
                   "share's ticker - use ticker searcher."
        if len(tickers_list) > 5 and username not in top_users:
            return 'No more than 5 papers. If you want more services contact us'
        for ticker in tickers_list:
            if not models.check_ticker(ticker):
                return f'{ticker} ticker not found at our base. You may search for ticker of your company with /search ' \
                       f'command.'

        corr_df = cc(tickers_list)
        df_styled = corr_df.style.background_gradient()
        corr_path = f'{home_path}PNGs/corr_df_{dt.now().strftime("%H%M%S%d%m%Y")}.png'
        dfi.export(df_styled, corr_path, table_conversion='matplotlib')
        corr_path = corr_path
        return corr_path


    def welcoming_buttons():
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton(text='Portfolio', callback_data='/model_cb'),
                   types.InlineKeyboardButton(text='Correlations', callback_data='/corr_calc_cb'),
                   types.InlineKeyboardButton(text='Find ticker', callback_data='/searcher_cb'),
                   types.InlineKeyboardButton(text='About', callback_data='/about'),
                   types.InlineKeyboardButton(text='Contact us', callback_data='/contact'))
        return markup


    # TODO need to add function with initialising of handle\cursor and execute it inside
    def log_message(message, msg_type):
        """
        Function to log messages in db
        :param message: incoming message from user
        :param msg_type: callback button
        :return:
        """
        cursor, conn = initialize_cursor()
        query = f"INSERT INTO LOGGING VALUES ('{dt.now().strftime('%H:%M:%S %d-%m-%Y')}', '{message.chat.id}', " \
                f"'{message.from_user.username}', '{message.text}', '{msg_type}')"
        print(query)
        cursor.execute(query)
        conn.commit()


    # TODO test it
    # TODO separate bot which sends updates from logging functions
    def log_fall(traceback_str):
        cursor, conn = initialize_cursor()
        query = f"INSERT INTO falls (TimeStamp, Traceback) VALUES ('{dt.now().strftime('%H:%M:%S %d-%m-%Y')}'," \
                f"'{replace_quotes(traceback.format_exc())}')"
        print(query)
        cursor.execute(query)
        conn.commit()


    # TODO test it
    def initialize_cursor():
        """
        Function to initialize connection to db for logging
        Depend on STAND_TYPE (TEST\PROD)
        :return: cursor
        """
        if STAND_TYPE == 'TEST':
            conn = sqlite3.connect('sqlite_python.db')
            cursor = conn.cursor()
        else:
            DATABASE_URL = os.environ['DATABASE_URL']
            conn = psycopg2.connect(DATABASE_URL, sslmode='require')
            cursor = conn.cursor()
        return cursor, conn

    # TODO put here instrument to stop bot or turn on\ turn off
    # здесь нужно петлю нормально вывести чтобы не отключаться из-за падения сети
    while 1:
        try:
            print('start telebot')
            bot.polling()
        except Exception as e:
            print(f'Exception:\n{e}\n\nTraceback:\n{traceback.format_exc()}')
            log_fall(str(traceback.format_exc()))
except Exception as e:
    print(f'Exception:\n{e}\n\nTraceback:\n{traceback.format_exc()}')
    log_fall(str(traceback.format_exc()))

# TODO инкапсулиповать корелляции

