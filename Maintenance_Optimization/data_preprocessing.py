import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import arviz as az
import json

def preprocess_maintenance_data():
    # Load and filter data
    log = pd.read_csv('processed_maintenance_log.csv')
    igbt_log = log[log['PROBLEMCODE'] == 'FANS']
    igbt_log = igbt_log.sort_values(by=['Inverter', 'Module'])
    igbt_log = igbt_log.reset_index(drop=True)

    # Process maintenance records
    maintenance_index = igbt_log[['Inverter', 'Module']].duplicated(keep=False)
    maintenance_log = igbt_log[maintenance_index].reset_index(drop=True)
    maintenance_log['REPORTDATE'] = pd.to_datetime(maintenance_log['REPORTDATE'])

    # Calculate lifespans
    maintenance_log_shift = maintenance_log.shift(1)
    replace_index = (maintenance_log[['Inverter', 'Module']] == 
                    maintenance_log_shift[['Inverter', 'Module']])
    rows_same_as_previous = replace_index.all(axis=1)
    date_diff = maintenance_log['REPORTDATE'] - maintenance_log_shift['REPORTDATE']
    maintenance_log.loc[rows_same_as_previous, 'LifeSpan'] = date_diff[rows_same_as_previous]
    replace_log = maintenance_log[rows_same_as_previous].reset_index(drop=True)

    return process_failure_data(replace_log)

def process_failure_data(replace_log):
    # Initialize failure log array
    failure_log = [[] for _ in range(24)]
    
    # Group failures by month
    for index, row in replace_log.iterrows():
        month_index = (row['REPORTDATE'].year - 2019) * 12 + row['REPORTDATE'].month - 5
        if 0 <= month_index < 24:
            failure_log[month_index].append(row['LifeSpan'].days/365.25)
    
    return train_weibull_parameters(failure_log)

def train_weibull_parameters(failure_log):
    # Initial parameters
    shape_prior, scale_prior = 0.76, 3.64
    
    # Train global parameters
    failure_ob = np.array([x for sublist in failure_log for x in sublist if x])
    
    with pm.Model() as model:
        shape = pm.Normal('shape', mu=shape_prior, sigma=0.001)
        scale = pm.Normal('scale', mu=scale_prior, sigma=0.4)
        weibull = pm.Weibull('weibull', alpha=shape, beta=scale, observed=failure_ob)
        trace = pm.sample(1000, tune=2000, return_inferencedata=False)
    
    # Train monthly parameters
    parameter_sc = np.zeros((24, 2))
    for day, failure in enumerate(failure_log):
        if failure:
            with pm.Model() as model:
                shape = pm.Normal('shape', mu=shape_prior, sigma=0.001)
                scale = pm.Normal('scale', mu=scale_prior, sigma=0.53)
                weibull = pm.Weibull('weibull', alpha=shape, beta=scale, 
                                   observed=np.array(failure))
                trace = pm.sample(1000, tune=2000, return_inferencedata=False)
                parameter_sc[day][0] = np.mean(trace['shape'])
                parameter_sc[day][1] = np.mean(trace['scale'])
        else:
            parameter_sc[day][0] = shape_prior
            parameter_sc[day][1] = scale_prior
    
    # Save parameters to file
    params = {
        'global_shape': float(shape_prior),
        'global_scale': float(scale_prior),
        'monthly_parameters': parameter_sc.tolist()
    }
    
    with open('weibull_parameters.json', 'w') as f:
        json.dump(params, f)
    
    return params

if __name__ == "__main__":
    preprocess_maintenance_data() 