import sys
from pandas import DataFrame

default_values: dict = {
    'CANCELLED': 0,
    'DIVERTED': 0
}

columns_to_remove_for_canceled: list[str] = [
    'DIVERTED',  # the flight has been diverted to an unplanned airport
]

columns_to_remove_for_diverted: list[str] = [
    'CANCELLED',  # the flight has been cancelled
]


def preprocess_for_canceled(data: DataFrame) -> DataFrame:

    # Remove useless columns
    data.drop(columns_to_remove_for_canceled, axis=1, inplace=True)

    return data

def preprocess_for_diverted(data: DataFrame) -> DataFrame:

    # Remove useless columns
    data.drop(columns_to_remove_for_diverted, axis=1, inplace=True)

    return data


def preprocess(data: DataFrame) -> DataFrame:

    data = common_preprocess(data)
    if sys.argv[1] == "canceled":
        return preprocess_for_canceled(data)
    if sys.argv[1] == "diverted":
        return preprocess_for_diverted(data)


def common_preprocess(data: DataFrame) -> DataFrame:

    # Replace Nan values with the correct default values
    data.fillna(default_values, inplace=True)

    # Remove rows with Nan key values
    data.dropna(how='any', axis='index', inplace=True)

    return data


# Division between training set and testing set

# Chiedere cosa fare in caso di valori null su colonne possibilmente rilevanti
# Chiedere se i dati su delay causati da cose come aereo in ritardo o meteo sono disponibili al momento del calcolo

# Trasformare tutto in numeri

# Normalizzare i dati tra -1 e 1 (?)
