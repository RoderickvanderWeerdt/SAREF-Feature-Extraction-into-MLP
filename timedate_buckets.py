import datetime as dt
import pandas as pd

def create_buckets_dict(year,month,day,hour,minute,second,weekday):
	bucket_dict = {}
	if year: bucket_dict['year'] = []
	if month: bucket_dict['month'] = []
	if day: bucket_dict['day'] = []
	if hour: bucket_dict['hour'] = []
	if minute: bucket_dict['minute'] = []
	if second: bucket_dict['second'] = []
	if weekday: bucket_dict['weekday'] = []
	return bucket_dict


def add_buckets(table_fn, new_fn, timestamp_columnname, year=True,month=True,day=True,hour=True,minute=True,second=True,weekday=True):
	data = pd.read_csv(table_fn)
	all_timestamps = data[timestamp_columnname]
	buckets = create_buckets_dict(year,month,day,hour,minute,second,weekday)
	for timestamp in all_timestamps:
		print(timestamp)
		timestamp = dt.datetime.fromisoformat(timestamp)
		if year: buckets['year'].append(timestamp.year)
		if month: buckets['month'].append(timestamp.month)
		if day: buckets['day'].append(timestamp.day)
		if hour: buckets['hour'].append(timestamp.hour)
		if minute: buckets['minute'].append(timestamp.minute)
		if second: buckets['second'].append(timestamp.second)
		if weekday: buckets['weekday'].append(timestamp.weekday())

	for key in buckets.keys():
		data[key] = buckets[key]

	data.to_csv(new_fn, index=False)


if __name__ == '__main__':
	# table_filename = "weather_SMALL_data.csv"
	# new_filename = "weather_SMALL_data_w_buckets.csv"
	table_filename = "res2_weather_SMALL_table.csv"
	new_filename = "res2_weather_SMALL_buckets_table.csv"

	# table_filename = "weather_data_1hr_w_timestamp.csv"
	# new_filename = "weather_data_1hr_w_buckets.csv"
	column_name = "timestamp"
	add_buckets(table_filename, new_filename, column_name)