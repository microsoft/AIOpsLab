# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import time
from datetime import datetime, timedelta, timezone
from ssl import create_default_context
from enum import Enum
from typing import Union
from kubernetes import client

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionTimeout

from . import monitor_config, root_path, get_pod_list, get_services_list
from .utils.extract import merge_csv


class LogAPI:
    def __init__(self, url: str, username: str, password: str):
        if monitor_config["es_use_cert"] == "True":
            context = create_default_context(cafile=monitor_config["es_cert_path"])
            self.elastic = Elasticsearch(
                [url],
                basic_auth=(username, password),
                # timeout=60,
                max_retries=5,
                retry_on_timeout=True,
                ssl_context=context,
            )
        else:
            self.elastic = Elasticsearch(
                [url],
                basic_auth=(username, password),
                verify_certs=False,
                # timeout=60,
                max_retries=5,
                retry_on_timeout=True,
            )
        self.log_pod_list, self.service_list = self.initialize_pod_and_service_lists()

    def initialize_pod_and_service_lists(self, custom_namespace=None):
        namespace = custom_namespace or monitor_config["namespace"]
        v1 = client.CoreV1Api()
        pod_list = [
            pod
            for pod in get_pod_list(v1, namespace=namespace)
            if not pod.startswith("loadgenerator-") and not pod.startswith("redis-cart")
        ]
        service_list = get_services_list(v1, namespace=namespace)
        return pod_list, service_list

    def log_extract(self, start_time=None, end_time=None, path=None):
        time_interval = 5 * 60
        csv_list = []
        os.makedirs(path, exist_ok=True)
        while start_time < end_time:
            current_end_time = start_time + time_interval
            if current_end_time > end_time:
                current_end_time = end_time
            data = self.log_extract_(start_time=start_time, end_time=current_end_time)
            if len(data) != 0:
                # Export
                data.to_csv(
                    f"{path}/log-{start_time}_{current_end_time}.csv", index=False
                )
                csv_list.append(f"{path}/log-{start_time}_{current_end_time}.csv")
            start_time = current_end_time
            time.sleep(1)

        if not csv_list:
            print("No logs found for the given time range.")
        else:
            merge_csv(path, csv_list, f"log_{int(time.time())}")

    # log data export
    def log_extract_(self, start_time=None, end_time=None):
        quert_size = 7500

        indices = self.elastic.indices.get(index="logstash-*")
        indices = choose_index_template(indices, start_time, end_time)
        print("indices", indices)

        if isinstance(start_time, int):
            # start_time = datetime.fromtimestamp(start_time)
            start_time = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        if isinstance(end_time, int):
            # end_time = datetime.fromtimestamp(end_time)
            end_time = datetime.fromtimestamp(end_time, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        # print(start_time, end_time)
        query = {
            "size": quert_size,
            "query": {
                "range": {"@timestamp": {"gte": start_time, "lte": end_time}}
                # "bool": {
                #     "must": [
                #         {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}}
                #     ]
                # }
            },
            # "sort": ["_doc"],
            "sort": [{"@timestamp": {"order": "asc"}}],
        }
        data = []

        st_time = time.time()
        for index in indices:
            try:
                page = self.elastic.search(index=index, body=query, scroll="15s")
                # page = self.elastic.search(index="logstash-*", body=query, scroll="15s")
                # print("Query being sent to Elasticsearch:")
                # print(json.dumps(query, indent=2))
                # print("Elasticsearch response:", page)
                # print(query)
                data.extend(page["hits"]["hits"])
                # scroll_id = page["_scroll_id"]

                # while True:
                #     page = self.elastic.scroll(scroll_id=scroll_id, scroll="15s")
                #     hits_len = len(page["hits"]["hits"])
                #     data.extend(page["hits"]["hits"])
                #     if hits_len < quert_size:
                #         break
                #     scroll_id = page["_scroll_id"]
            except ConnectionTimeout as e:
                print("Connection Timeout:", e)

        print("search time: ", time.time() - st_time)
        st_time = time.time()

        # TODO: extract data
        # data = log_processing_online_boutique(data)
        data = log_processing_hotel_reservation(data)
        print("process time:", time.time() - st_time)
        return data

    def get_log_number_by_day(self, time_select):
        data = []
        try:
            indices = self.elastic.indices.get(index="logstash-*")

            logs_per_day = {}  # store log per day

            # ONE_DAY aggregate according to hour
            if time_select == TimeSelect.ONE_DAY:
                for index in indices:
                    response = self.elastic.count(index=index)
                    index_date_str = index.split("-")[-1]  # get the date
                    index_date = datetime.strptime(index_date_str, "%Y.%m.%d.%H")
                    # if within the last one day
                    if index_date >= datetime.now() - timedelta(days=1):
                        # day_key = index_date.strftime("%Y-%m-%d")

                        if index_date not in logs_per_day:
                            logs_per_day[index_date] = 0
                        logs_per_day[index_date] += response["count"]
            elif time_select == TimeSelect.ONE_WEEK:
                for index in indices:
                    response = self.elastic.count(index=index)
                    index_date_str = index.split("-")[-1]
                    index_date = datetime.strptime(index_date_str, "%Y.%m.%d.%H")

                    # if within the last seven days
                    if index_date >= datetime.now() - timedelta(days=7):
                        day_key = index_date.strftime(
                            "%Y-%m-%d"
                        )  # transform the datetime to %Y-%m-%d format string as key

                        if index_date not in logs_per_day:
                            logs_per_day[day_key] = 0
                        logs_per_day[day_key] += response["count"]
            elif time_select == TimeSelect.TWO_WEEK:
                for index in indices:
                    response = self.elastic.count(index=index)
                    index_date_str = index.split("-")[-1]  # get the date
                    index_date = datetime.strptime(
                        index_date_str, "%Y.%m.%d.%H"
                    )  # transform to datetime

                    # if within two weeks
                    if index_date >= datetime.now() - timedelta(days=14):
                        day_key = index_date.strftime(
                            "%Y-%m-%d"
                        )  # transform the datetime to %Y-%m-%d format string as key

                        if index_date not in logs_per_day:
                            logs_per_day[day_key] = 0
                        logs_per_day[day_key] += response["count"]
            else:
                print(f"Wrong input params: {time_select}")
                return data
            for date_str, log_count in logs_per_day.items():
                data.append({"date": date_str, "log_count": log_count})
        except ConnectionTimeout as e:
            print("Connection Timeout:", e)
        return data

    def query(
        self, start_time: Union[int, datetime, str], end_time: Union[int, datetime, str]
    ):
        if isinstance(start_time, str):
            start_time = int(start_time)
        if isinstance(end_time, str):
            end_time = int(end_time)

        # get the indices from the time span
        indices = self.elastic.indices.get(index="logstash-*")
        indices = choose_index_template(indices, start_time, end_time)

        start_time = datetime.fromtimestamp(start_time)
        end_time = datetime.fromtimestamp(end_time)

        query_size = 2500
        # Elasticsearch query
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}}
                    ]
                }
            },
            "sort": {"@timestamp": {"order": "asc"}},
            "size": query_size,
        }
        # return Elasticsearch query result
        data = []

        for index in indices:
            try:
                page = self.elastic.search(index=index, body=query, scroll="15s")
                data.extend(page["hits"]["hits"])
                scroll_id = page["_scroll_id"]

                while True:
                    page = self.elastic.scroll(scroll_id=scroll_id, scroll="15s")
                    hits_len = len(page["hits"]["hits"])
                    data.extend(page["hits"]["hits"])
                    if hits_len < query_size:
                        break
                    scroll_id = page["_scroll_id"]
            except ConnectionTimeout as e:
                print("Connection Timeout:", e)
        data = log_for_query_filter(data)
        print("len data", len(data))
        return data


def message_extract(json_str):
    message = json_str
    try:
        if "severity" in json_str:
            data = json.loads(json_str)
            message = "".join(
                ["severity:", data["severity"], ",", "message:", data["message"]]
            )
        elif "level" in json_str:
            data = json.loads(json_str)
            message = "".join(
                ["level:", data["level"], ",", "message:", data["message"]]
            )
    except:
        pass
    return message


def log_processing_hotel_reservation(logs):
    log_id_list = []
    ts_list = []
    date_list = []
    pod_list = []
    ms_list = []
    container_name_list = []
    namespace_list = []
    node_name_list = []

    for log in logs:
        try:
            # Extract information from the log
            log_id = log["_id"]
            timestamp = log["_source"]["@timestamp"]
            pod_name = log["_source"]["kubernetes"]["pod"]["name"]
            container_name = log["_source"]["kubernetes"]["container"]["name"]
            namespace = log["_source"]["kubernetes"]["namespace"]
            node_name = log["_source"]["kubernetes"]["node"]["name"]
            message = log["_source"]["message"]

            # Convert timestamp to a readable format
            timestamp_obj = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            timestamp_unix = timestamp_obj.timestamp()

        except KeyError as e:
            print(f"KeyError encountered: {e}")
            print(f"Skipping log due to missing fields: {log}")
            continue

        # Append to the respective lists
        log_id_list.append(log_id)
        ts_list.append(timestamp_unix)
        date_list.append(timestamp)
        pod_list.append(pod_name)
        container_name_list.append(container_name)
        namespace_list.append(namespace)
        node_name_list.append(node_name)
        ms_list.append(message)

    # Create DataFrame
    dt = pd.DataFrame(
        {
            "log_id": log_id_list,
            "timestamp": ts_list,
            "date": date_list,
            "pod_name": pod_list,
            "container_name": container_name_list,
            "namespace": namespace_list,
            "node_name": node_name_list,
            "message": ms_list,
        }
    )

    return dt


def log_processing_online_boutique(logs):
    log_id_list = []
    ts_list = []
    date_list = []
    pod_list = []
    ms_list = []
    for log in logs:
        try:
            cmdb_id = log["_source"]["kubernetes"]["pod"]["name"]
            if cmdb_id not in self.log_pod_list:
                continue
            timestamp = log["_source"]["@timestamp"]
            timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            timestamp = timestamp.timestamp()
            format_ts = log["_source"]["@timestamp"]
            message = message_extract(log["_source"]["message"])
        except Exception as e:
            continue
        log_id_list.append(log["_id"])
        pod_list.append(cmdb_id)
        date_list.append(format_ts)
        ts_list.append(timestamp)
        ms_list.append(message)
    dt = pd.DataFrame(
        {
            "log_id": log_id_list,
            "timestamp": ts_list,
            "date": date_list,
            "cmdb_id": pod_list,
            "message": ms_list,
        }
    )
    return dt


def log_for_query_filter(logs):
    filtered_log = []
    for log in logs:
        try:
            cmdb_id = log["_source"]["kubernetes"]["pod"]["name"]
            if cmdb_id not in self.log_pod_list:
                continue
        except Exception as e:
            continue
        filtered_log.append(log)
    return filtered_log


def choose_index_template(indices, start_time, end_time):
    indices_template = set()
    for index in indices:
        date_str = ".".join(index.split("-")[1].split(".")[:-1])
        indices_template.add("logstash-" + date_str + "*")

    start_datetime_utc = datetime.fromtimestamp(start_time)
    end_datetime_utc = datetime.fromtimestamp(end_time)

    dates_in_range = set()
    current_datetime = start_datetime_utc

    while current_datetime <= end_datetime_utc:
        dates_in_range.add("logstash-" + current_datetime.strftime("%Y.%m.%d") + "*")
        current_datetime += timedelta(days=1)

    dates_in_range.add("logstash-" + end_datetime_utc.strftime("%Y.%m.%d") + "*")

    selected_patterns = indices_template.intersection(dates_in_range)
    print(f"selected_patterns: {selected_patterns}")
    return selected_patterns


class TimeSelect(Enum):
    ONE_DAY = 1
    ONE_WEEK = 2
    TWO_WEEK = 3

    @classmethod
    def get_item_by_value(cls, enum_type, value):
        for member in enum_type.__members__.values():
            if member.value == int(value):
                return member
        return ValueError(f"no member found with value : {value}")


if __name__ == "__main__":
    logger = LogAPI(
        monitor_config["api"], monitor_config["username"], monitor_config["password"]
    )
    # end_time = datetime.now(timezone.utc)
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5)
    print(int(start_time.timestamp()), int(end_time.timestamp()))
    save_path = root_path / "log_output"
    os.makedirs(save_path, exist_ok=True)
    logger.log_extract(
        start_time=int(start_time.timestamp()),
        end_time=int(end_time.timestamp()),
        path=save_path,
    )
