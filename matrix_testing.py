import numpy as np

def fromAccumulateToDifference(arr):
	arr0 = np.pad(arr, (1,0), 'constant')[:-1]
	return arr-arr0

def process_file(data_fn, new_fn):
	data = np.genfromtxt(data_fn, dtype=None, delimiter=',', names=True)
	
	data = data[np.argsort(data[:])] #sort on timestamp
	for col_name in data.dtype.names:
		if col_name != "timestamp":
			try:
				data[col_name] = fromAccumulateToDifference(data[col_name])
			except:
				print("The column \"", col_name, "could not be changed from AccumulateToDifference")
	data = data[1:] #remove the first instance, because there was no previous instance to determine the difference.
	np.random.shuffle(data) #shuffle the data
	
	headers = ''
	with open(data_fn, 'r') as datafile:
		headers = datafile.readline()[:-1]
	
	fmt = ['%s'] + ['%1.3f']*(len(headers.split(','))-1)
	np.savetxt(new_fn, data, fmt=fmt, delimiter=',', newline='\n', header=headers, comments="")


if __name__ == '__main__':
	# data_fn = "new_table.csv"
	# data_fn = "german_table_LARGE.csv"
	data_fn = "res2_subset_data.csv"
	new_fn = "res2_subset_table.csv"
	process_file(data_fn, new_fn)

	# a = np.array([["2015-06-11T21:00:00+00:00", 1, 2],["2017-03-12T14:00:00+00:00", 3, 4]])
	# b = np.array([["2016-03-12T14:00:00+00:00", 5, 6]])
	# c = np.append(a, b, axis=0)
	# print(c)
	# print(c[1][0])
	# sorted_c = c[np.argsort(c[:, 0])]
	# print(sorted_c)
