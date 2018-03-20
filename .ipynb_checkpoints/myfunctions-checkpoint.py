import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation

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
