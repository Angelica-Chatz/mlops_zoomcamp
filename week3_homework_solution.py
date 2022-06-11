from datetime import datetime

import pandas as pd

import pickle

""" from prefect import flow, get_run_logger, task
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule """

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


""" def log_task(task_msg):
    logger = get_run_logger()
    logger.info(task_msg) """

#@task
def get_paths(date=None):

    path_prefix = "./data/fhv_tripdata_"
    file_type = ".parquet"

    if date:
        date_conv = datetime.strptime(date, "%Y-%m-%d")
        input_year = str(date_conv.year)
        val_month = f"{(date_conv.month - 1):02d}"
        train_month =  f"{(date_conv.month - 2):02d}"
    else:
        input_year = str(datetime.now().year)
        val_month = f"{(datetime.now().month - 1):02d}"
        train_month =  f"{(datetime.now().month - 2):02d}"

    val_path = path_prefix + input_year + '-' + val_month + file_type
    train_path = path_prefix + input_year + '-' + train_month + file_type

    #log_task(f"{train_path}, {val_path}")
    print(f"{train_path}, {val_path}")

    return train_path, val_path


#@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

#@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        #log_task(f"The mean duration of training is {mean_duration}")
        print(f"The mean duration of training is {mean_duration}")
    else:
        #log_task(f"The mean duration of validation is {mean_duration}")
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

#@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values
    
    #log_task(f"The shape of X_train is {X_train.shape}")
    print(f"The shape of X_train is {X_train.shape}")
    
    #log_task(f"The DictVectorizer has {len(dv.feature_names_)} features")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    #log_task(f"The MSE of training is: {mse}")
    print(f"The MSE of training is: {mse}")
    return lr, dv

#@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    #log_task(f"The MSE of validation is: {mse}")
    print(f"The MSE of validation is: {mse}")
    return

#@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    train_path, val_path = get_paths(date)

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical)#.result()
    run_model(df_val_processed, categorical, dv, lr)

    with open(f"models/dv-{date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    with open(f"models/model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)

#main()
main(date="2021-03-15")


# prefect deployment
""" DeploymentSpec(
    name="model_training_prefect",
    flow_location="/home/deva/mlops_zoomcamp/week3_homework_solution.py",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["prefect_hw3"]
) """
