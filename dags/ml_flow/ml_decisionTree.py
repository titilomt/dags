import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import argparse

from urllib.parse import urlparse

parser = argparse.ArgumentParser()

parser.add_argument("inputPath", help="arquivo input iris", type=str)
parser.add_argument("experiment", help="experimento MLflow", type=str)
parser.add_argument("modelName", help="nome modelo MLflow", type=str)
parser.add_argument("max_depth", help="parametro max_depth",
                    default=0, type=int)
parser.add_argument(
    "random_state", help="parametro random_state", default=0, type=int)

args = parser.parse_args()

iris = pd.read_csv(args.inputPath, sep=",")

X_dict = iris.iloc[:, 0: (iris.shape[1] - 2)].T.to_dict().values()
vect = DictVectorizer(sparse=False)

le = LabelEncoder()
y_train = le.fit_transform(iris.iloc[:, (iris.shape[1] - 2)])

X_train = vect.fit_transform(X_dict)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, random_state=args.random_state, test_size=0.1)

try:
    idExperiment = mlflow.create_experiment(args.experiment)
except:
    idExperiment = mlflow.get_experiment_by_name(args.experiment).experiment_id

with mlflow.start_run(experiment_id=idExperiment):
    tree = DecisionTreeClassifier(
        random_state=args.random_state, criterion='gini', max_depth=3)
    tree = tree.fit(X_train, y_train)

    print("Acuracia:", tree.score(X_train, y_train))

    y_pred = tree.predict(X_test)

    mlflow.log_metric("accuracy_score", accuracy_score(y_test, y_pred))
    print("Acurácia de previsão:", accuracy_score(y_test, y_pred))

    # Model registry does not work with file store

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(
            tree, "model", registered_model_name=args.modelName)
    else:
        mlflow.sklearn.log_model(tree, "model")
