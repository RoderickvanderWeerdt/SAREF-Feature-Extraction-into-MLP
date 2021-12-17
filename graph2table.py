import rdflib
import numpy as np

#MAKE SURE TIMESTAMPS ARE ALL OF THE SAME FORMAT!!!

def save_to_file(new_table_name, collected_values, header):
	with open(new_table_name, 'w') as table_file:
		table_file.write(','.join(header)+'\n')
		for k in collected_values.keys():
			if 'None' in collected_values[k]:
				continue
			# print(collected_values[k])
			table_file.write(k+','+','.join(collected_values[k])+'\n')

def open_table_file(table_name):
	data_dict = {}
	with open(table_name, 'r') as table_file:
		header = table_file.readline()[:-1].split(',')
		for line in table_file.readlines():
			line = line[:-1].split(',')
			data_dict[line[0]] = line[1:]
	return header, data_dict


def get_graph(graph_name, loaded_graph):
	g = rdflib.Graph()
	if graph_name != "" and loaded_graph==None:
		print("\nloading graph '"+graph_name+"'...")
		g.parse(graph_name)
	elif loaded_graph != None:
		print("\nfound pre-loaded graph")
		g = loaded_graph
	else:
		print("ERROR, YOU NEED TO SUPPLY A GRAPH")

	g.namespace_manager.bind('saref', 'https://saref.etsi.org/core/')
	g.namespace_manager.bind('ic', 'https://interconnectproject.eu/example/')

	print("found", len(g), "triples in the graph.")
	return g

def get_value_from_graph(graph, query, device_URI, device_i, collected_values, feature_vector_row):
	qres = graph.query(query, initBindings={"device": device_URI})

	for row in qres:
		timestamp = str(row.t)
		if '+' in row.t:
			timestamp = row.t[:-6] #remove DST from notation
		if row.val == '':
			continue
		try:
			collected_values[timestamp][device_i] = str(float(row.val))
		except:
			collected_values[timestamp] = feature_vector_row[:]
			try:
				collected_values[timestamp][device_i] = str(float(row.val))
			except:
				print("THIS VALUE GAVE AN ERROR:", row.val)

	return collected_values


def get_values_from_graph(graph_name, device_names, loaded_graph=None, new_table_name="new_table.csv"):
	g = get_graph(graph_name,loaded_graph)

	q = """SELECT ?val ?t WHERE {
		?device saref:makesMeasurement ?measurement .
		?measurement saref:hasValue ?val .
		?measurement saref:hasTimestamp ?t .
		}"""

	collected_values = {}
	n_features = len(device_names)
	feature_vector_row = ['None']*n_features
	header = ["timestamp"] + feature_vector_row[:]
	for device_name, device_i in zip(device_names,range(0,n_features)):
		header[device_i+1] = device_name
		device_URI = rdflib.URIRef(device_name)
		print("quering for device:", device_URI)
		collected_values = get_value_from_graph(g, q, device_URI, device_i, collected_values, feature_vector_row)

	save_to_file(new_table_name, collected_values, header)
	print("created a new table with", len(collected_values.keys()), "lines.")

def add_values_from_graph(graph_name, device_names, table_name, new_table_name="new_table.csv", loaded_graph=None):
	g = get_graph(graph_name,loaded_graph)

	q = """SELECT ?val ?t WHERE {
		?device saref:makesMeasurement ?measurement .
		?measurement saref:hasValue ?val .
		?measurement saref:hasTimestamp ?t .
		}"""

	collected_values = {}
	n_features = len(device_names)
	feature_vector_row = ['None']*n_features
	header = ["timestamp"] + feature_vector_row[:]
	for device_name, device_i in zip(device_names,range(0,n_features)):
		header[device_i+1] = device_name
		device_URI = rdflib.URIRef(device_name)
		print("quering for device:", device_URI)
		collected_values = get_value_from_graph(g, q, device_URI, device_i, collected_values, feature_vector_row)

	original_header, data_dict = open_table_file(table_name)
	print(original_header)
	print(header)

	for timestamp in data_dict.keys():
		try:
			data_dict[timestamp] += collected_values[timestamp]
		except:
			data_dict[timestamp] += feature_vector_row

	new_header = original_header + header [1:] #without the second timestamp

	save_to_file(new_table_name, data_dict, new_header)
	print("created a new table with", len(data_dict.keys()), "lines.")


def get_all_device_names(graph_name="", loaded_graph=None):
	g = get_graph(graph_name, loaded_graph)

	print("quering to find all devices")

	qres = g.query("""SELECT ?device WHERE {
		?prop saref:isMeasuredByDevice ?device .
		}""")

	devices = []
	for row in qres:
		devices.append(row.device)

	print("found", len(devices), "different devices")
	return devices, g


if __name__ == '__main__':
	# EXAMPLE 1, creating a new table dataset from a graph:
	graph_name = "res2_kg_subset.ttl"
	new_table_fn = "res2_subset_data.csv"
	dev_names = ["https://interconnectproject.eu/example/DEKNres2_GI","https://interconnectproject.eu/example/DEKNres2_WM","https://interconnectproject.eu/example/DEKNres2_CP","https://interconnectproject.eu/example/DEKNres2_FR","https://interconnectproject.eu/example/DEKNres2_DW"]
	g = None
	print(dev_names)
	get_values_from_graph(graph_name, dev_names, g, new_table_fn)

	# EXAMPLE 2, adding datapoints from a different graph to an existing table dataset, based on matching timestamps
	graph_name = "weather_temp_graph_SMALL.ttl"
	table_fn = "res2_subset_data.csv"
	new_table_fn = "weather_SMALL_data.csv"
	dev_names, g = get_all_device_names(graph_name) #"https://interconnectproject.eu/example/Konstanz_TempC"

	add_values_from_graph(graph_name, dev_names, table_fn, new_table_fn, g)

	# graph_name = "german_data_LARGE.nt"
	# new_table_fn = "german_table_LARGE.csv"
	# # dev_names = ["https://interconnectproject.eu/example/DEKNres2_GI","https://interconnectproject.eu/example/DEKNres2_WM","https://interconnectproject.eu/example/DEKNres2_CP","https://interconnectproject.eu/example/DEKNres2_FR","https://interconnectproject.eu/example/DEKNres2_DW"]
	# dev_names, g = get_all_device_names(graph_name) #"https://interconnectproject.eu/example/Konstanz_TempC"
	# # g = None
	# print(dev_names)
	# get_values_from_graph(graph_name, dev_names, g, new_table_fn)

	# graph_name = "weather_temp_graph.ttl"
	# table_fn = new_table_fn
	# new_table_fn = "weather_data.csv"
	# dev_names, g = get_all_device_names(graph_name) #"https://interconnectproject.eu/example/Konstanz_TempC"

	# add_values_from_graph(graph_name, dev_names, table_fn, new_table_fn, g)
