import sys
from time import sleep

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import xgboost as xgb

import psycopg2
import psycopg2.extras # cursor

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


# Try to connect
conn = psycopg2.connect(host='192.168.0.201', user='bartek', password='Aga', database='logs', port=5432)
conn.set_client_encoding('UTF8')
cur = conn.cursor()


logs_df = pd.read_sql_query('select * from log_data', conn)
print(logs_df.shape)
logs_df.head()


print(logs_df.shape)

features = ['rank_w_ip' , 'avg_id_rows_current' , 'id_parity' , 'rank_w_ip_parity' , 'the_same_parity']
print(features)


target = 'target_1'
y = logs_df[target]
X = logs_df[features]
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=324)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

models = {}

lrs = [0.1, 0.01, 0.001, 0.0001, 2.5, 0.25, 0.025, 0.0025, 0.5, 0.05, 0.005]
reg_alphas = [0, 0.2, 0.5, 0.8, 1]
reg_lambdas = [0, 0.2, 0.5, 0.8, 1]
base_scores = [0.4, 0.5, 0.6]

model_id = 0
combinations = len(lrs) * len(reg_lambdas) * len(reg_alphas) * len(base_scores) + model_id

for lr in lrs:
    for reg_alpha in reg_alphas:
        for reg_lambda in reg_lambdas:
            for base_score in base_scores:
                model_id += 1
                print("fitting model {} / {}".format(model_id, combinations))

                xgb_model = xgb.XGBClassifier(base_score=base_score, booster='gbtree', colsample_bylevel=1,
                                              colsample_bytree=1, gamma=0, learning_rate=lr, max_delta_step=0,
                                              max_depth=6, min_child_weight=1, missing=None, n_estimators=1000,
                                              n_jobs=-1, nthread=None, objective='binary:logistic', random_state=0,
                                              reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=1, seed=None,
                                              silent=True, subsample=1)

                xgb_model = xgb_model.fit(X=X_train, y=y_train, verbose=1)
                train_accuracy = xgb_model.score(X=X_train, y=y_train)
                test_accuracy = xgb_model.score(X=X_test, y=y_test)

                print("model_id_{}: test_accuracy={} (lr={}, reg_alpha={}, reg_lambda={}, base_score={})" \
                      .format(model_id, test_accuracy, lr, reg_alpha, reg_lambda, base_score))

                models[model_id] = {
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "lr": lr,
                    "reg_alpha": reg_alpha,
                    "reg_lambda": reg_lambda,
                    "base_score": base_score
                }



# Najlepszy model:
# model_id_520: test_accuracy=0.9964164841728052 (lr=0.025, reg_alpha=1, reg_lambda=0.8, base_score=0.4)

# Wyniki do pandas:
xgb_models_df = pd.DataFrame(models).transpose()
xgb_models_df.head()


# max z różnych kolumn
xgb_models_df.max()
xgb_models_df.test_accuracy.max()

# najlepszy model w zbiorze
xgb_models_df[xgb_models_df.test_accuracy >= 0.996610]


# zapis do csv UWAGA!
xgb_models_df.to_csv("./xgb_models2.csv") # Żeby NIE NADPISAĆ !!!

# Oryginalnie: xgb_models_df = pd.DataFrame(models).transpose()
# LUB wczytać z csv:
# wczytanie csv
xgb_models_df = pd.read_csv("./xgb_models2.csv")
xgb_models_df.head()


# zapis do bazy PostgreSQL

from sqlalchemy import create_engine

engine = create_engine("postgresql://bartek:Aga@192.168.0.201:5432/logs")
xgb_models_df.to_sql("xgb_models", engine)

# Best Model:
xgb_model = xgb.XGBClassifier(base_score=0.6, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.025, max_delta_step=0,
       max_depth=6, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=-1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=1, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

xgb_model = xgb_model.fit(X=X_train, y=y_train, verbose=2)
xgb_model.score(X=X_test, y=y_test)

xgb.plot_importance(xgb_model)
xgb.plot_tree(xgb_model, num_trees=524)

xgb.to_graphviz(xgb_model, num_trees=524)




# Próby z transakcją / może się przydać
# Transaction: begin transaction + try + commit/rollback

conn = psycopg2.connect(host='192.168.0.201', user='bartek', password='Aga', database='logs', port=5432)
conn.set_client_encoding('UTF8')
cur = conn.cursor()

# start transaction
cur.execute("begin transaction")

try:
    for k, v in models.items():
        cur.execute("""
            insert into xgb_models_results (
                python_model_id,
                lr,
                test_accuracy
            ) values ({}, {}, {})
        """.format(k, v['lr'], v['test_accuracy']))
    conn.commit()
    print("All records inserted, commit")
except:
    conn.rollback()
    print("Records NOT inserted, rollback!")



