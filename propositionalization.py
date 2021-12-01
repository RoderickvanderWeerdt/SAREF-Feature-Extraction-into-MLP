import rdflib
import numpy as np

#MAKE SURE TIMESTAMPS ARE ALL OF THE SAME FORMAT!!!

def get_graph(graph_name, loaded_graph):
	g = rdflib.Graph()
	if graph_name != "" and loaded_graph==None:
		print("loading graph...")
		g.parse(graph_name)
	elif loaded_graph != None:
		print("found pre-loaded graph")
		g = loaded_graph
	else:
		print("ERROR, YOU NEED TO SUPPLY A GRAPH")

	g.namespace_manager.bind('saref', 'https://saref.etsi.org/core/')
	g.namespace_manager.bind('ic', 'https://interconnectproject.eu/example/')

	print("found", len(g), "triples in the graph.")
	return g

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

		qres = g.query(q, initBindings={"device": device_URI})

		for row in qres:
			if row.val == '':
				continue
			try:
				collected_values[str(row.t)][device_i] = str(float(row.val))
			except:
				collected_values[str(row.t)] = feature_vector_row[:]
				try:
					collected_values[str(row.t)][device_i] = str(float(row.val))
				except:
					print("THIS VALUE GAVE AN ERROR:", row.val)


	with open(new_table_name, 'w') as table_file:
		table_file.write(','.join(header)+'\n')
		for k in collected_values.keys():
			if 'None' in collected_values[k]:
				continue
			# print(collected_values[k])
			table_file.write(k+','+','.join(collected_values[k])+'\n')

	print("created a new table with", len(collected_values.keys()), "lines.")

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

# dev_names = ["https://interconnectproject.eu/example/DEKNres1_FR", "https://interconnectproject.eu/example/DEKNres1_DW"]
# get_values_from_graph("german_data_.ttl", dev_names, None, "new_table.csv")

# graph_name = "german_data_.ttl"
graph_name = "german_data_LARGE.nt"
graph_name = "res2_kg_subset.ttl"

new_table_fn = "res2_subset_data.csv"

# dev_names, g = get_all_device_names(graph_name)
#removed wm_6, because it had to many empty measurements
dev_names = ["https://interconnectproject.eu/example/DEKNres2_GI","https://interconnectproject.eu/example/DEKNres2_WM","https://interconnectproject.eu/example/DEKNres2_CP","https://interconnectproject.eu/example/DEKNres2_FR","https://interconnectproject.eu/example/DEKNres2_DW"]
# dev_names = ["https://interconnectproject.eu/example/DEKNres2_GI","https://interconnectproject.eu/example/DEKNres3_GE","https://interconnectproject.eu/example/DEKNres4_GE","https://interconnectproject.eu/example/DEKNres3_GI","https://interconnectproject.eu/example/DEKNres3_CP","https://interconnectproject.eu/example/DEKNres3_WM","https://interconnectproject.eu/example/DEKNres3_FR","https://interconnectproject.eu/example/DEKNres5_DW","https://interconnectproject.eu/example/DEKNres5_RF","https://interconnectproject.eu/example/DEKNres3_RF","https://interconnectproject.eu/example/DEKNres2_WM","https://interconnectproject.eu/example/DEKNres6_DW","https://interconnectproject.eu/example/DEKNres1_GI","https://interconnectproject.eu/example/DEKNres6_FR","https://interconnectproject.eu/example/DEKNres4_PV","https://interconnectproject.eu/example/DEKNres4_WM","https://interconnectproject.eu/example/DEKNres1_HP","https://interconnectproject.eu/example/DEKNres1_WM","https://interconnectproject.eu/example/DEKNres1_PV","https://interconnectproject.eu/example/DEKNres3_DW","https://interconnectproject.eu/example/DEKNres4_DW","https://interconnectproject.eu/example/DEKNres6_GE","https://interconnectproject.eu/example/DEKNres3_PV","https://interconnectproject.eu/example/DEKNres1_FR","https://interconnectproject.eu/example/DEKNres5_GI","https://interconnectproject.eu/example/DEKNres6_GI","https://interconnectproject.eu/example/DEKNres2_CP","https://interconnectproject.eu/example/DEKNres6_PV","https://interconnectproject.eu/example/DEKNres2_FR","https://interconnectproject.eu/example/DEKNres2_DW","https://interconnectproject.eu/example/DEKNres6_CP","https://interconnectproject.eu/example/DEKNres5_WM","https://interconnectproject.eu/example/DEKNres4_HP","https://interconnectproject.eu/example/DEKNres1_DW","https://interconnectproject.eu/example/DEKNres4_FR","https://interconnectproject.eu/example/DEKNres4_GI","https://interconnectproject.eu/example/DEKNres4_RF","https://interconnectproject.eu/example/DEKNres4_EV"]
g = None
print(dev_names)
get_values_from_graph(graph_name, dev_names, g, new_table_fn)
