import numpy as np
import pandas as pd
from settings import db_creds

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

import psycopg2
import psycopg2.extras # cursor
from sqlalchemy import create_engine

import matplotlib.pyplot as plt
import seaborn as sns

import graphviz
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus



# %matplotlib inline
sns.set_style("darkgrid")


# Try to connect: remote database
conn = psycopg2.connect(**db_creds)
conn.set_client_encoding('UTF8')
cur = conn.cursor()


# Prace z wynikami modeli deep learning
# ######################################


### Uwaga! tylko modele BEZ early stopping i bez super-długiego uczenia!
dl_models_df = pd.read_sql_query('select * from v_dl_models_performance where run_id <> 8 and epochs < 32000', conn)
dl_models_df.head()

# Histogram i inne wizualizacje wyników
mask = dl_models_df["test_accuracy"] >= 0.85

dl_models_df[mask].hist("test_accuracy", bins=100, figsize=(12, 6))

dl_models_df.boxplot("test_accuracy", by="run_id", figsize=(12, 6))

dl_models_df[mask].plot("epochs", "test_accuracy", figsize=(12, 6), kind="scatter", alpha=0.2)


plt.figure(figsize=(12, 6))
sns.regplot(y="test_accuracy", x="epochs", data=dl_models_df[mask])

plt.figure(figsize=(12, 6))
sns.regplot(y="test_accuracy", x="epochs", data=dl_models_df[mask], x_estimator=np.mean)

dl_models_df[mask].plot("lr", "test_accuracy", figsize=(12, 6), kind="scatter", alpha=0.2)


# Uwaga run_id 700 <- kolor! Rekod do 9
def recode_val_run_id(a):
    if a != 700:
        return a
    else:
        return 9


dl_models_df["run_id_recoded"] = dl_models_df.run_id.apply(lambda r: recode_val_run_id(r))
dl_models_df[mask].plot.scatter(x="epochs", y="test_accuracy", c="batch_size", cmap="coolwarm",
                                figsize=(12, 6), alpha=0.8)


dl_models_df[mask].plot.scatter(x="epochs", y="test_accuracy", c="epochs", cmap="coolwarm", figsize=(12, 6), alpha=0.8)

plt.figure(figsize=(12, 6))
sns.scatterplot(x="epochs", y="test_accuracy", hue="run_id_recoded", data=dl_models_df[mask])


# Heatmap dla hyperparameters
# Tylko test accuracy, loss + hyperparameters
dl_hyperparameters = dl_models_df.loc[:, ["test_accuracy", "test_loss", "lr", "batch_size", "epochs"]]

plt.figure(figsize=(10, 8))
sns.heatmap(dl_hyperparameters.corr(), cmap="coolwarm", square=True, annot=True)


## Drzewko decyzyjne: Deep Learning hyperparameters' performance
## -------------------------------------------------------------



dl_dtr = DecisionTreeRegressor(criterion='mse', max_depth=8, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=10,
           min_samples_split=30, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best')


X_dl_hyperparams = dl_hyperparameters.iloc[:, 2:]
X_dl_hyperparams.head()


### Próbnie: Standaryzacja y żeby zobaczyc lepiej różnice
### StandardScaler tylko do macierzy, więc sam sobie napisałem niżej do standaryzacji

def scaler(x):
    m = np.mean(x)
    s = np.std(x)

    return np.divide((x-m), s)


# y_scaled = StandardScaler().fit_transform(y=dl_models_df["test_accuracy"])
y_scaled = scaler(dl_models_df["test_accuracy"])


# domyślnie na surpwych wartościach test_accuracy
dl_dtr_model = dl_dtr.fit(X=X_dl_hyperparams, y=dl_models_df["test_accuracy"])


# Jakie hyperparameters / jakie feature_importances_
for n, i in zip(X_dl_hyperparams.columns, dl_dtr_model.feature_importances_):
    print("{0} importance: {1:.8f}".format(n, i))


# Wizualizacja drzewka dla wyników modeli deep learning
# ------------------------------------------------------


# Moje udoskonalenie: https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176

from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = export_graphviz(dl_dtr_model, out_file=None,
                filled=True, rounded=True,
                feature_names=X_dl_hyperparams.columns,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


### Jeszcze żeby zapisać jako pdf z innego tutoriala:
# https://scikit-learn.org/stable/modules/tree.html<br>
# Uwaga!<br>
# graph = graphviz.Source(dot_data) jest INNYM OBIEKTEM niż wcześniej<br> graph = pydotplus.graph_from_dot_data(dot_data)!!!<br>


# https://scikit-learn.org/stable/modules/tree.html
# Uwaga! Tu graph jest INNYM OBIEKTEM !!!
graph2 = graphviz.Source(dot_data)

# zapisanie pliku pdf z wizualizacja drzewka
graph2.render("dl_models_hyperparameters_tree")



# Prace z wynikami modeli xgboost
# ###############################


# Oryginalnie: xgb_models_df = pd.DataFrame(models).transpose()
# LUB wczytać z csv:
# zapis do csv
xgb_models_df = pd.read_csv("./xgb_models.csv")
xgb_models_df.head()


# najlepsze test_accuracy
xgb_models_df.test_accuracy.max()

xgb_models_df[xgb_models_df.test_accuracy == xgb_models_df.test_accuracy.max()]


# zapis do csv
# xgb_models_df.to_csv("./xgb_models.csv")


# zapis do bazy PostgreSQL

from sqlalchemy import create_engine

# JUŻ ZROBIONE !!!
# engine = create_engine("postgresql://bartek:Aga@192.168.0.201:5432/logs")
# xgb_models_df.to_sql("xgb_models", engine)

## Winner is:

# ale wcześniej aby było mozna go wytrenować jeszcze raz pobranie i przygotowanie danych:

conn = psycopg2.connect(host='192.168.0.201', user='bartek', password='Aga', database='logs', port=5432)
conn.set_client_encoding('UTF8')
cur = conn.cursor()

logs_df = pd.read_sql_query('select * from akuratne_25k', conn)
print(logs_df.shape)
logs_df.head()

features = ['rank_w_ip' , 'avg_id_rows_current' , 'id_parity' , 'rank_w_ip_parity' , 'the_same_parity']
features

target = 'target_1'
y = logs_df[target]

X = logs_df[features]
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=324)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


xgb_model = xgb.XGBClassifier(base_score=0.6, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.025, max_delta_step=0,
       max_depth=6, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=-1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=1, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

xgb_model = xgb_model.fit(X=X_train, y=y_train, verbose=2)
xgb_model.score(X=X_test, y=y_test)

# test_accuracy: 0.9966155683854271


# xgb plots:
# ----------

# feature importance
xgb.plot_importance(xgb_model)

# xgb tree
xgb.plot_tree(xgb_model, num_trees=524)

# wielkie na ubuntulaptop i nie mieści się na ekranie ale to może być kwestia rozdzielczości
xgb.to_graphviz(xgb_model, num_trees=5)


# wizualizacje
# -------------

# histogram
xgb_models_df.hist("test_accuracy", bins=7, figsize=(12, 6))

xgb_models_df.boxplot("test_accuracy", by="lr", figsize=(12, 6))

xgb_models_df.plot("lr", "test_accuracy", figsize=(12, 6), kind="scatter", alpha=0.2)

xgb_models_df.plot("lr", "test_accuracy", figsize=(12, 6), kind="scatter", alpha=0.2, c="test_accuracy", cmap="coolwarm")

xgb_models_df.plot("lr", "base_score", figsize=(12, 6), kind="scatter", alpha=0.2, c="test_accuracy", cmap="coolwarm")

xgb_models_df.plot("model_id", "test_accuracy", figsize=(12, 6), kind="scatter", alpha=0.2)

xgb_models_df.plot("model_id", "test_accuracy", figsize=(12, 6), kind="scatter", alpha=0.2, c="lr", colormap="coolwarm")

xgb_models_df.boxplot("test_accuracy", by="reg_alpha", figsize=(12, 6))

xgb_models_df.boxplot("test_accuracy", by="reg_lambda", figsize=(12, 6))

plt.figure(figsize=(10, 8))
sns.heatmap(xgb_models_df.iloc[:, 1:].corr(), cmap="coolwarm", square=True, annot=True)


## Drzewko decyzyjne: xgboost hyperparameters' performance
## ---------------------------------------------------------



## Drzewko decyzyjne: xgboost hyperparameters' performancefrom sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(criterion='mse', max_depth=8, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=20,
           min_samples_split=40, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best')


dtr_model = dtr.fit(X=xgb_models_df.loc[:, ["base_score", "lr", "reg_alpha", "reg_lambda"]], y=xgb_models_df["test_accuracy"])


dtr_model.feature_importances_

xgb_models_df[["base_score", "lr", "reg_alpha", "reg_lambda"]].head()


### Narysowanie drzewka

# https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(dtr_model, out_file=dot_data,
                filled=True, rounded=True,
                feature_names=["base_score", "lr", "reg_alpha", "reg_lambda"],
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# Alternatywnie plot tree! Dodatkowo można zapisac do pdf graph.render()
# https://scikit-learn.org/stable/modules/tree.html

import graphviz
from sklearn.tree import export_graphviz


dot_data = export_graphviz(dtr_model, out_file=None,
                     feature_names=["base_score", "lr", "reg_alpha", "reg_lambda"],
                     filled=True, rounded=True,
                     special_characters=True)

graph = graphviz.Source(dot_data)


# tworzy PDF!
#graph.render("xgb_models_hyperparameters_tree")
graph


# UWAGA!
# dot_data NIE musi być dot_data = StringIO()
# bo także działa z dot_data = export_graphviz(....) z tej drugiej metody

# dot_data NIE musi być dot_data = StringIO() bo także działa z dot_data = export_graphviz(....)
# wtedy już bez dot_data.values()

graph2 = pydotplus.graph_from_dot_data(dot_data)
Image(graph2.create_png())

# TODO: dodać pozostałe hyperparameters modeli, szczególnie: n_estimators i regularyzacja)
