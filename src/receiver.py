# Load libraries

import json
import os
import sys
global data
import time

i = 0
i1 = 0
i2 = 3
i3 = 5
i4 = 7
i5 = 9
i6 = 1
i7 = 4
i8 = 6
i9 = 8

while i < 1000:
    i1 = i1+1
    i2 = i2 +1
    i3 = i3 +1
    i4 = i4 +1
    i5 = i5 +1
    i6 = i6 +1
    i7 = i7 +1
    i8 = i8 +1
    i9 = i9 +1
    j = 0

    print("LIve data for Graph %d " %i)


    jsonObject = {}
    #jsonObject['EngineID'] = "df_solution['Engine_ID'].values.tolist()"
    jsonObject['Predicted'] = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i1, i2, i3, i4, i5, i6, i7, i8, i9]
    jsonObject['time'] = [j+1, j+2, j+3, j+4, j+5, j+6, j+7, j+8, j+9, j+10, j+11, j+12, j+13, j+14, j+15, j+16, j+17, j+18 ]
    #jsonObject['Actual_RUL'] = df_solution['Actual_RUL'].values.tolist()
    
    #jsonName = str("../Output/") + aName + str("_") + cleanApproach + str("_Predicted.json")
    with open('./assets/json/dummy.json', 'w') as outfile:
        json.dump(jsonObject, outfile)

    print(jsonObject)

    i = i+1
    j = j+1

    if i == 1000:
        i = 0
        i1 = 0
        i2 = 3
        i3 = 5
        i4 = 7
        i5 = 9
        i6 = 1
        i7 = 4
        i8 = 6
        i9 = 8
    time.sleep(3)
