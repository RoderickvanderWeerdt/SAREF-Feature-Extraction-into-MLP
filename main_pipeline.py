from graph2table import get_all_device_names, get_values_from_graph, add_values_from_graph
from accumulate2diff import accumulate2diff
from timedate_buckets import add_buckets
from opds_data_mlp import perform_prediction

# starting files
graph_file = "res2_kg_subset.ttl"
extra_graph_file = "weather_temp_graph_SMALL.ttl"

# files created during the process
table_file = "res2_subset_data.csv"
table_w_extra_file = "res2_weather_SMALL_data.csv"
table_with_diff = "res2_weather_SMALL_table.csv"
table_with_buckets = "res2_weather_SMALL_buckets_table.csv"


## EXAMPLE 1, creating a new table dataset from a graph:
dev_names = ["https://interconnectproject.eu/example/DEKNres2_GI","https://interconnectproject.eu/example/DEKNres2_WM","https://interconnectproject.eu/example/DEKNres2_CP","https://interconnectproject.eu/example/DEKNres2_FR","https://interconnectproject.eu/example/DEKNres2_DW"]
g = None
get_values_from_graph(graph_file, dev_names, g, table_file)

## EXAMPLE 2, adding datapoints from a different graph to an existing table dataset, based on matching timestamps
extra_dev_names, g = get_all_device_names(extra_graph_file) #"https://interconnectproject.eu/example/Konstanz_TempC"
add_values_from_graph(extra_graph_file, extra_dev_names, table_file, table_w_extra_file, g)

## step 2, changing the values from accumulated values to difference since last hour.
accumulate2diff(table_w_extra_file, table_with_diff, skip_columns=extra_dev_names) #skip weather column

## step 3, create timestamp buckets, based on the timestamp in the dataset.
column_name = "timestamp"
add_buckets(table_with_diff, table_with_buckets, column_name)

## step 4, performing the prediction of <target_dev> with specific devices <dev_list>
dev_list = ['https://interconnectproject.eu/example/DEKNres2_CP','https://interconnectproject.eu/example/DEKNres2_WM','https://interconnectproject.eu/example/DEKNres2_FR']
target_dev = 'https://interconnectproject.eu/example/DEKNres2_GI'
perform_prediction(table_with_buckets, dev_list, target_dev)

