import string
import download
import read
import pandas as pd

download.DownloadData()
data:pd.DataFrame = read.GetFirstFrame()

#Remove the empty columns
data.dropna(how='all', axis='columns', inplace=True)

#Remove conlumns not considered useful for computation of cancelled flights:
# - WHEELS_OFF: moment when the aircraft takes off
# - WHEELS_ON: moment when the aircraft land
# - TAXI_IN: time required to taxi from the landing strip (wheels_on) to the terminal
# - TAXI_OUT: time required to taxi from the terminal to the landing strip(wheels off)
# - OP_CARRIER_FL_NUM: flight number assigned from the carrier
# - DEP_TIME: actual time of departure
# - DEP_DELAY: departure delay
# - ARR_TIME: actual time of arrival
# - ARR_DELAY: delay on arrival
# - DIVERTED: the flight has been diverted to an unplanned airport
# - ACTUAL_ELAPSED_TIME: time from taxi_in to taxi_out
# - AIR_TIME: time between wheels_off and wheels_on

columns_to_remove:list[
    'WHEELS_OFF':string,
    'WHEELS_ON':string,
    'TAXI_IN':string,
    'TAXI_OUT':string,
    'OP_CARRIER_FL_NUM':string,
    'DEP_TIME':string,
    'DEP_DELAY':string,
    'ARR_TIME':string,
    'ARR_DELAY':string,
    'DIVERTED':string,
    'ACTUAL_ELAPSED_TIME':string,
    'AIR_TIME':string
]

data.drop(columns_to_remove, axis=1, inplace=True)

print(data)
