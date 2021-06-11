import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
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
X_train, X_test, y_train, y_test = train_test_split(iris.drop(
    columns=['class', 'classEncoder']), iris['class'], test_size=0.1, random_state=args.random_state)


try:
    idExperiment = mlflow.create_experiment(args.experiment)
except:
    idExperiment = mlflow.get_experiment_by_name(args.experiment).experiment_id


with mlflow.start_run(experiment_id=idExperiment):
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("random_state", args.random_state)

    clf = RandomForestClassifier(
        max_depth=args.max_depth, random_state=args.random_state)
    clf.fit(X_train, y_train)
    predictCLF = clf.predict(X_test)

    mlflow.log_metric("accuary_score", accuracy_score(y_test, predictCLF))

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(
            clf, "model", registered_model_name=args.modelName)
    else:
        mlflow.sklearn.log_model(clf, "model")
