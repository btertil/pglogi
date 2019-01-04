# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:07:31 2019

@author: bondi
"""

import sys
from time import sleep
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Softmax, Activation, Dropout
from keras.activations import relu
from keras.initializers import VarianceScaling
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical
from keras.wrappers import scikit_learn
from keras.callbacks import EarlyStopping

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import psycopg2
import psycopg2.extras  # cursor

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
# %%

# Try to connect
try:
    conn = psycopg2.connect(host='192.168.0.101', user='bartek', password='Aga', database='logs', port=5432)
    # utf8
    conn.set_client_encoding('UTF8')
    # cursor
    cur = conn.cursor()
except:
    print("I am unable to connect to the database.")
    conn = None
    cur = None
    sys.exit(1)


# %%

logs_df = pd.read_sql_query('select * from akuratne_25k', conn)
print(logs_df.shape)
logs_df.head()

print(logs_df.shape)

features = ['rank_w_ip', 'avg_id_rows_current', 'id_parity', 'rank_w_ip_parity', 'the_same_parity']
print(features)

target = 'target_1'
print(target)

X = logs_df[features]
X.head()

y = logs_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=324)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = StandardScaler().fit(X_train)

# std scaler jest szkolony tylko na train
# wszelkie przetwarzanie testu na podstawie regół, znalezioych dla train, także dla pre-process
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
def fit_and_evaluate_model(models, model_id=0, lr=0.001, batch_size=1024, epochs=3500, patience=None):
    # Keras model
    k_model = Sequential()
    # 1st layer
    # k_model.add(Dense(units=512, kernel_initializer=VarianceScaling, input_shape=X.shape[0], activation=None))
    k_model.add(Dense(128, activation=None, input_shape=(5,)))
    k_model.add(BatchNormalization())
    k_model.add(Activation("relu"))
    k_model.add(Dropout(rate=0.1))
    # 2nd layer
    k_model.add(Dense(64, activation=None))
    k_model.add(BatchNormalization())
    k_model.add(Activation("relu"))
    k_model.add(Dropout(rate=0.1))
    # 3nd layer
    k_model.add(Dense(16, activation=None))
    k_model.add(BatchNormalization())
    k_model.add(Activation("relu"))
    # 4nd layer
    k_model.add(Dense(1, activation=None))
    k_model.add(BatchNormalization())
    k_model.add(Activation("sigmoid"))

    k_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=lr), metrics=["accuracy"])


    # Train
    history = k_model.fit(X_train_scaled, y=y_train.values, epochs=epochs,
                          batch_size=batch_size, validation_split=0.25, verbose=0)

    # Plot training history
    def plot_accuracy_and_loss(trained_model, test_accuracy):

        validation = False

        hist = trained_model.history
        acc = hist['acc']
        loss = hist['loss']
        try:
            val_acc = hist['val_acc']
            val_loss = hist['val_loss']
            validation = True
        except KeyError as e:
            print("No validation data defined, showing only training set hostory")
        epochsw = range(1, len(acc) + 1)

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        plt.suptitle("Model_id={0}: lr={1}, batch_size={2}, epochs={3}, test_accuracy={4: .10f}" \
                     .format(model_id, lr, batch_size, epochs, test_accuracy), color='grey', fontsize=20)

        ax[0].tick_params(colors="grey")
        ax[0].plot(epochsw, acc, 'g', label='Training accuracy')
        if validation:
            ax[0].plot(epochsw, val_acc, 'r', label='Validation accuracy')
        ax[0].set_ylabel('Accuracy', color='grey', fontsize=12)
        ax[0].set_title('Training and validation accuracy', color='grey', fontsize=16)
        ax[0].set_xlabel('Epochs', color='grey', fontsize=12)

        ax[0].legend()
        ax[0].grid(True)

        ax[1].tick_params(colors="grey")
        ax[1].plot(epochsw, loss, 'g', label='Training cost')
        if validation:
            ax[1].plot(epochsw, val_loss, 'r', label='Validation cost')
        ax[1].legend()
        ax[1].set_title('Training and validation loss', color='grey', fontsize=16)
        ax[1].set_xlabel('Epochs', color='grey', fontsize=12)
        ax[1].set_ylabel('Loss', color='grey', fontsize=12)
        ax[1].grid(True)

        plt.show()

    # Evaluate & Plot
    score_test = k_model.evaluate(X_test_scaled, y_test, verbose=0)
    test_loss = score_test[0]
    test_accuracy = score_test[1]

    # Print evaluation results on test dataset
    print("model_id_{}: lr={}, batch_size={}, epochs={}".format(model_id, lr, batch_size, epochs))
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)

    # Zapisanie do dictionary
    models["{}".format(model_id)] = {
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

    # insert statement
    sql_statement = """
                insert into dl_models (python_model_id, lr, batch_size, epochs, test_loss, test_accuracy)
                 values ({}, {}, {}, {}, {}, {})
            """.format(model_id, lr, batch_size, epochs, test_loss, test_accuracy)

    global conn
    global cur

    try:
        cur.execute(sql_statement)
        conn.commit()

    except Exception:
        print("Re-connecting to the database...")
        try:
            cur.close()
            conn.close()
        except:
            print("Previous connection not available")

        # Wait 5s and try re-establish connection
        sleep(3)

        try:
            conn = psycopg2.connect(host='192.168.0.101', user='bartek', password='Aga', database='logs', port=5432)
            conn.set_client_encoding('UTF8')
            cur = conn.cursor()
            print("New connection has been established")
        except:
            print("Unable to connect to the database.")
            sleep(180)
            try:
                conn = psycopg2.connect(host='192.168.0.101', user='bartek', password='Aga', database='logs', port=5432)
                conn.set_client_encoding('UTF8')
                cur = conn.cursor()
                print("New connection has been established")
            except:
                print("I am unable to connect to the database, quitting...")
                sys.exit()

        cur.execute(sql_statement)
        conn.commit()

    # Best model?
    current_best = models.get("best", None)
    if current_best is not None and current_best["test_accuracy"] >= test_accuracy:
        pass

    else:
        # aktualizacja best
        models["best"] = {
            "model_id": model_id,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }
        # nowy wykres z historią uczenia
        plot_accuracy_and_loss(history, test_accuracy)


# %%
models = {}

# %%

# lrs = [10, 1, 0.1, 0.01, 0.001, 0.0001]
lrs = [0.0007]
batch_sizes = [4096]
epochss = [i for i in range(15000, 17000, 50)]


model_id = 817
kombinacje = len(lrs) * len(epochss) * len(batch_sizes) + model_id

for lr in lrs:
    for epochs in epochss:
        for batch_size in batch_sizes:

            model_id += 1
            print("\n\nFitting model {} / {}".format(model_id, kombinacje))
            fit_and_evaluate_model(models, model_id, lr=lr, batch_size=batch_size, epochs=epochs)

        print("\n\nBest model:")
        print(models["best"])

