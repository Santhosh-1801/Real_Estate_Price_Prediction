import json
import pickle
import numpy as n

__locations=None
__data_columns=None
__model=None

def get_estimated_prices(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index= -1
    x = n.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)
def get_location_names():
    return __locations
def get_data_columns():
    return __data_columns

def load_saved_artifacts():
    print("loading saved artifacts..start")
    global __data_columns
    global __locations


    with open("./artifacts/colinformation.json","r") as p:
        __data_columns=json.load(p)['data_columns']
        __locations=__data_columns[3:]

    global __model
    with open("./artifacts/home_price_prediction_lr.pickle","rb") as q:
        __model=pickle.load(q)
    print("loading saved artifacts..done")
if __name__=="__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_prices('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_prices('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_prices('Kalhalli', 1000, 2, 2))
    print(get_estimated_prices('Ejipura', 1000, 2, 2))