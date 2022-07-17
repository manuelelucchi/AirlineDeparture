import sys
from pandas import DataFrame

default_values: dict = {
    'CANCELLED': 0,
    'DIVERTED': 0
}

columns_to_remove_for_canceled: list[str] = [
    'WHEELS_OFF',  # moment when the aircraft takes off
    'WHEELS_ON',  # moment when the aircraft land
    # time required to taxi from the landing strip (wheels_on) to the terminal
    'TAXI_IN',
    # time required to taxi from the terminal to the landing strip(wheels off)
    'TAXI_OUT',
    'OP_CARRIER_FL_NUM',  # flight number assigned from the carrier
    'DEP_TIME',  # actual time of departure
    'DEP_DELAY',  # departure delay
    'ARR_TIME',  # actual time of arrival
    'ARR_DELAY',  # delay on arrival
    'DIVERTED',  # the flight has been diverted to an unplanned airport
    'ACTUAL_ELAPSED_TIME',  # time from taxi_in to taxi_out
    'AIR_TIME'  # time between wheels_off and wheels_on
]


def preprocess_for_canceled(data: DataFrame) -> DataFrame:

    # Remove useless columns
    data.drop(columns_to_remove_for_canceled, axis=1, inplace=True)

    return data


def preprocess(data: DataFrame) -> DataFrame:

    data = common_preprocess(data)
    if sys.argv[1] == "canceled":
        return preprocess_for_canceled(data)
    else:
        raise NotImplementedError()


def common_preprocess(data: DataFrame) -> DataFrame:

    # Remove the empty columns
    data.dropna(how='all', axis='columns', inplace=True)

    # Replace Nan values with the correct default values
    data.fillna(value=default_values, axis='columns', inplace=True)

    # Remove rows with Nan key values
    data.dropna(how='any', axis='index', inplace=True)

    return data


# Division between training set and testing set

# Chiedere cosa fare in caso di valori null su colonne possibilmente rilevanti

# Trasformare tutto in numeri

# Normalizzare i dati tra -1 e 1 (?)
