# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pathlib
import sys
import pytz
from datetime import datetime, timedelta

from kubernetes import config, client
from yaml import full_load

root_path = pathlib.Path(__file__).parent
sys.path.append(root_path)
# read the configuration file
with open(root_path / "monitor_config.yaml", "r") as f:
    monitor_config = full_load(f)


# root_config = full_load(open(root_path / "config.yaml", "r"))
def get_pod_list(v1, namespace="default"):
    pod_list = v1.list_namespaced_pod(namespace)
    pod_names = []
    # list the names of all Pods and store their names in the list
    for pod in pod_list.items:
        pod_names.append(pod.metadata.name)
    return pod_names


def get_services_list(v1, namespace="default"):
    # get all of the list under certain namespace
    service_list = v1.list_namespaced_service(namespace)

    services_names = []
    for service in service_list.items:
        services_names.append(service.metadata.name)

    return services_names


config.kube_config.load_kube_config(config_file=monitor_config["kubernetes_path"])
v1 = client.CoreV1Api()

# pod_list = [
#     pod
#     for pod in get_pod_list(v1, namespace=monitor_config["namespace"])
#     if not pod.startswith("loadgenerator-") and not pod.startswith("redis-cart")
# ]

# service_list = get_services_list(v1, namespace=monitor_config["namespace"])

# from observor.metric_api import PrometheusAPI
# # from observor.trace_api import TraceAPI
# from observor.log_api import logAPI
# prom = PrometheusAPI(monitor_config["prometheusApi"])

# Define time range for exporting metrics
# end_time = datetime.now()
# start_time = end_time - timedelta(minutes=10)


# Define the save path for metrics
# save_path = root_path / "metrics_output"

# prom.export_all_metrics(start_time=start_time, end_time=end_time, save_path=str(save_path), step=10)

# trace = TraceAPI(monitor_config['api'], monitor_config['username'], monitor_config['password'])
# log = logAPI(monitor_config['api'], monitor_config['username'], monitor_config['password'])
