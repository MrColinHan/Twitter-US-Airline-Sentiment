import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set(style=”whitegrid”, palette=”pastel”, color_codes=True)
#sns.mpl.rc(“figure”, figsize=(10,6))

'''
    problems with the original clean up process and address validation process: 
        1. not all numbers are invalid, e.g. zip code is also a valid address
        2. locations written in a different languages could still be US addresses
        3. locations seem to be wrong could still be correct address
            e.g. 'YYC' is abbreviation for an airport
        4. Google geocoding API only allows 2500 free lookups each day. Difficult to use it to check 
           whether address is a valid US address
'''