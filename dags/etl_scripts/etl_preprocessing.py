import pandas as pd

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("inputPath", help="arquivo input Iris", type=str)
parser.add_argument("outputPath", help="arquivo output Iris", type=str)
parser.add_argument("errorPath", help="arquivo log Iris", type=str)

args = parser.parse_args()

dfIris = pd.read_csv(args.inputPath)


def is_between(a, x, b):
    return min(a, b) <= x <= max(a, b)


def _dataset_sanitize():
    dfIrisSanitize = dfIris.copy()
    dfIrisSanitize['error_message'] = False

    def __validator(row):
        if not is_between(4.3, row['sepal_length'], 7.9):
            row['error_message'] = 'sepal_length de valor {} fora do intervalo [4.3, 7.9]'.format(
                row['sepal_length'])
        elif not is_between(2.0, row['sepal_width'], 4.4):
            row['error_message'] = 'sepal_width de valor {} fora do intervalo [2.0, 4.4]'.format(
                row['sepal_width'])
        elif not is_between(1.0, row['petal_length'], 6.9):
            row['error_message'] = 'petal_length de valor {} fora do intervalo [1.0, 6.9]'.format(
                row['petal_length'])
        elif not is_between(0, row['classEncoder'], 2):
            row['error_message'] = 'classEncoder de valor {} fora do intervalo [0, 2]'.format(
                row['classEncoder'])
        elif row['class'] not in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
            row['error_message'] = "class de valor {} não existe em ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']".format(
                row['class'])

        return row

    dfIrisSanitize = dfIrisSanitize.apply(__validator, axis=1)

    dfIrisError = dfIrisSanitize.drop(
        dfIrisSanitize[dfIrisSanitize["error_message"] == False].index)

    dfIrisError.to_csv(args.errorPath, index=False)

    dfIrisSanitize.drop(
        dfIrisSanitize[dfIrisSanitize["error_message"] != False].index, inplace=True)

    dfIrisSanitize.drop(['error_message'], axis='columns', inplace=True)

    return dfIrisSanitize


df_sanitized = _dataset_sanitize()

df_sanitized.to_csv(args.outputPath, index=False)
