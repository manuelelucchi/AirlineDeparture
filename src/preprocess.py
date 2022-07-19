import datetime as dt
import sys
from numpy import ndarray
import numpy
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

names_columns_to_convert: list[str] = [
    'OP_CARRIER',
    'ORIGIN',
    'DEST',
]

date_columns_to_convert: list[str] = [
    'FL_DATE'
]

time_columns_to_convert: list[str] = [
    'CRS_DEP_TIME',
    'CRS_ARR_TIME',
    'CRS_ELAPSED_TIME'
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

    convert_names_into_numbers(data)
    convert_dates_into_numbers(data)
    convert_times_into_numbers(data)

    return data

def convert_names_into_numbers(data: DataFrame) -> DataFrame:

    for c in names_columns_to_convert:
        unique_values:ndarray = []
        values_map:dict = {}
        counter:float = 0

        unique_values = data[c].unique()
        unique_values = numpy.sort(unique_values)
        adder:float = 1 / len(unique_values)

        for v in unique_values:
            values_map[v] = counter
            counter += adder
        
        data[c].replace(to_replace=values_map, inplace=True)
    
    return data

def convert_dates_into_numbers(data: DataFrame) -> DataFrame:

    multiplier:float = 1 / 365

    for c in date_columns_to_convert:
        unique_values:ndarray = []
        values_map:dict = {}

        unique_values = data[c].unique()
        unique_values = numpy.sort(unique_values)
        
        for v in unique_values:
            date = dt.datetime.strptime(v, "%Y-%m-%d")
            day = date.timetuple().tm_yday
            values_map[v] = day * multiplier
        
        data[c].replace(to_replace=values_map, inplace=True)

    return data

def convert_times_into_numbers(data: DataFrame) -> DataFrame:

    multiplier:float = 1 / 2359

    for c in time_columns_to_convert:
        unique_values:ndarray = []
        values_map:dict = {}

        unique_values = data[c].unique()
        unique_values = numpy.sort(unique_values)
        
        for v in unique_values:
            values_map[v] = v * multiplier
        
        data[c].replace(to_replace=values_map, inplace=True)

    return data


# Division between training set and testing set

# Chiedere cosa fare in caso di valori null su colonne possibilmente rilevanti
# Chiedere se i dati su delay causati da cose come aereo in ritardo o meteo sono disponibili al momento del calcolo

# Trasformare tutto in numeri

# Normalizzare i dati tra -1 e 1 (?)
