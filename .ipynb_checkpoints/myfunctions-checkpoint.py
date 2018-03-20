
import base64
import datetime
import time as t
import io
import codecs
import pickle
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import savefig
import urllib
import os

from rq import Queue
from worker import conn
import uuid

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation


# Holidays dataframe (static global variable)
holidays = pd.DataFrame({'holiday': ['New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day',
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day',
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day',
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day',
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day',
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day',
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day',
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day',
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day',
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day',
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day',
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day',
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day',
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day',
                                     'Christmas Eve', 'Christmas Eve', 'Christmas Eve', 'Christmas Eve', 'Christmas Eve', 'Christmas Eve',
                                     'Christmas Eve', 'New Years Eve', 'New Years Eve', 'New Years Eve', 'New Years Eve', 'New Years Eve',
                                     'New Years Eve', 'New Years Eve', 'Valentines Day', 'Valentines Day', 'Valentines Day', 'Valentines Day',
                                     'Valentines Day', 'Valentines Day', 'Valentines Day', 'Halloween', 'Halloween', 'Halloween', 'Halloween',
                                     'Halloween', 'Halloween', 'Halloween', 'iPhone Launch', 'iPhone Launch', 'iPhone Launch', 'iPhone Launch',
                                     'iPhone Launch', 'iPhone Launch', 'Pixel Launch', 'Pixel Launch', 'Easter', 'Easter', 'Easter', 'Easter',
                                     'Easter', 'Easter', 'Easter', 'Samsung Launch', 'Samsung Launch', 'Samsung Launch', 'Samsung Launch',
                                     'Samsung Launch'],
                         'ds': pd.to_datetime(['2014-01-01', '2014-01-20', '2014-02-17', '2014-05-26', '2014-07-04', '2014-09-01',
                                               '2014-10-13', '2014-11-11', '2014-11-27', '2014-11-28', '2014-12-25', '2015-01-01',
                                               '2015-01-19', '2015-02-16', '2015-05-25', '2015-07-03', '2015-09-07', '2015-10-12',
                                               '2015-11-11', '2015-11-26', '2015-11-27', '2015-12-25', '2016-01-01', '2016-01-18',
                                               '2016-02-15', '2016-05-30', '2016-07-04', '2016-09-05', '2016-10-10', '2016-11-11',
                                               '2016-11-24', '2016-11-25', '2016-12-25', '2017-01-02', '2017-01-16', '2017-02-20',
                                               '2017-05-29', '2017-07-04', '2017-09-04', '2017-10-09', '2017-11-10', '2017-11-23',
                                               '2017-11-24', '2017-12-25', '2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28',
                                               '2018-07-04', '2018-09-03', '2018-10-08', '2018-11-12', '2018-11-22', '2018-11-23',
                                               '2018-12-25', '2019-01-01', '2019-01-21', '2019-02-18', '2019-05-27', '2019-07-04',
                                               '2019-09-02', '2019-10-14', '2019-11-11', '2019-11-28', '2019-11-29', '2019-12-25',
                                               '2020-01-01', '2020-01-20', '2020-02-17', '2020-05-25', '2020-07-03', '2020-09-07',
                                               '2020-10-12', '2020-11-11', '2020-11-26', '2020-11-27', '2020-12-25', '2014-12-24',
                                               '2015-12-24', '2016-12-24', '2017-12-24', '2018-12-24', '2019-12-24', '2020-12-24',
                                               '2014-12-31', '2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31',
                                               '2020-12-31', '2014-02-14', '2015-02-14', '2016-02-14', '2017-02-14', '2018-02-14',
                                               '2019-02-14', '2020-02-14', '2014-10-31', '2015-10-31', '2016-10-31', '2017-10-31',
                                               '2018-10-31', '2019-10-31', '2020-10-31', '2014-09-19', '2015-09-25', '2016-03-31',
                                               '2016-09-16', '2017-09-22', '2017-11-03', '2016-10-20', '2017-10-19', '2014-04-20',
                                               '2015-04-05', '2016-03-27', '2017-04-16', '2018-04-01', '2019-04-21', '2020-04-12',
                                               '2014-04-11', '2015-04-10', '2016-03-11', '2017-04-21', '2018-03-16'])})




# Main calculation script.
def calculation_script(data, forecast_periods):
    forecast_periods = forecast_periods
    cv_horizon = str(forecast_periods) + ' days'
    if data.shape[1] == 2:
        data['groups'] = 'None'
    data.columns = ['ds', 'y', 'groups']
    data_grouped = data.groupby('groups')
    groups = [group_name for group_name, group_data in data_grouped]
    models = [Prophet(holidays=holidays, holidays_prior_scale=4).fit(group_data) for group_name, group_data in data_grouped]
    futures = [model.make_future_dataframe(periods=forecast_periods) for model in models]
    forecasts = [model.predict(future) for model, future in zip(models, futures)]
    #plots = [model.plot(forecast) for model, forecast in zip(models, forecasts)]
    initials = [ str(group_data.shape[0] - ((forecast_periods * 2) + 2)) + ' days' for group_name, group_data in data_grouped]
    cv_results = [cross_validation(model, horizon=cv_horizon, period=cv_horizon, initial=initial) for model, initial in zip(models, initials)]
    for dataframe in cv_results:
        dataframe['error'] = dataframe['yhat'] - dataframe['y']
        dataframe['abs_error'] = dataframe['error'].transform(np.abs)
        dataframe['squared_error'] = dataframe['error'].transform(np.square)
        dataframe['pct_error'] = np.divide(dataframe['abs_error'], dataframe['y'])
    RMSE = [np.sqrt(np.mean(cv_result['squared_error'])) for cv_result in cv_results]
    MAPE = [np.mean(cv_result['pct_error']) for cv_result in cv_results]
    complete = pd.DataFrame(index = groups,
                        data = {'error_rmse': RMSE,
                                'error_mape': MAPE,
                                'models': models,
                                #'plots': plots,
                                'forecasts': forecasts})
    return codecs.encode(pickle.dumps(complete), "base64").decode() #The output table with forecast, plots, and errors by group, encoded as pickle.
