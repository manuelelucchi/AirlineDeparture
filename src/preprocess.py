import sys
from pandas import DataFrame

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
    # Remove the empty columns
    data.dropna(how='all', axis='columns', inplace=True)

    # Remove useless columns
    data.drop(columns_to_remove_for_canceled, axis=1, inplace=True)


def preprocess(data: DataFrame) -> DataFrame:
    if sys.argv[1] == "canceled":
        return preprocess_for_canceled(data)
    else:
        raise NotImplementedError()
