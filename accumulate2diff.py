import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def fromAccumulateToDifference(arr):
	arr0 = np.pad(arr, (1,0), 'constant')[:-1]
	return arr-arr0

def accumulate2diff(data_fn, new_fn, skip_columns):
	# data = np.genfromtxt(data_fn, dtype=None, delimiter=',', names=True)
	data = pd.read_csv(data_fn)

	
	# data = data[np.argsort(data[:])] #sort on timestamp
	data = data.sort_values("timestamp")
	# data = data.sort_values(b)
	headers = data.columns.values
	for col_name in headers:
		if col_name != "timestamp" and (not col_name in skip_columns):
			print(col_name)
			try:
				data[col_name] = fromAccumulateToDifference(data[col_name])
			except:
				print("The column \"", col_name, "could not be changed from AccumulateToDifference")
	data = data[1:] #remove the first instance, because there was no previous instance to determine the difference.

	data = shuffle(data) #shuffle the data
	data.to_csv(new_fn, index=False)



if __name__ == '__main__':
	# data_fn = "new_table.csv"
	# data_fn = "german_table_LARGE.csv"
	# data_fn = "res2_weather_SMALL_data.csv"
	# new_fn = "res2_weather_SMALL_table.csv"
	data_fn = "res2_data.csv"
	new_fn = "res2_table.csv"
	accumulate2diff(data_fn, new_fn, ['https://interconnectproject.eu/example/Konstanz_TempC'])

	# a = np.array([["2015-06-11T21:00:00+00:00", 1, 2],["2017-03-12T14:00:00+00:00", 3, 4]])
	# b = np.array([["2016-03-12T14:00:00+00:00", 5, 6]])
	# c = np.append(a, b, axis=0)
	# print(c)
	# print(c[1][0])
	# sorted_c = c[np.argsort(c[:, 0])]
	# print(sorted_c)
