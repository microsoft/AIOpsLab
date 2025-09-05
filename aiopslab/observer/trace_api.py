# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import socket
import time
import select
import subprocess
import threading
from datetime import datetime, timedelta

import requests
import pandas as pd

from aiopslab.observer import root_path


class TraceAPI:
    def __init__(self, namespace: str):
        self.port_forward_process = None
        self.namespace = namespace
        self.stop_event = threading.Event()
        self.output_threads = []

        if self.namespace == "astronomy-shop":
            # No NodePort in astronomy shop
            self.base_url = "http://localhost:16686/jaeger/ui"
            self.start_port_forward()
        else:
            # Other namespaces may expose a NodePort
            node_port = self.get_nodeport("jaeger", namespace)
            if node_port:
                self.base_url = f"http://localhost:{node_port}"
            else:
                self.base_url = "http://localhost:16686"
                self.start_port_forward()

    def get_nodeport(self, service_name, namespace):
        """Fetch the NodePort for the given service."""
        try:
            result = subprocess.check_output(
                [
                    "kubectl",
                    "get",
                    "service",
                    service_name,
                    "-n",
                    namespace,
                    "-o",
                    "jsonpath={.spec.ports[0].nodePort}",
                ],
                text=True,
            )
            nodeport = result.strip()
            print(f"NodePort for service {service_name}: {nodeport}")
            return nodeport
        except subprocess.CalledProcessError as e:
            print(f"Error getting NodePort: {e.output}")
            return None

    def print_output(self, stream):
        """Thread function to print output from a subprocess stream non-blockingly."""
        while not self.stop_event.is_set():
            # Check if there is content to read
            ready, _, _ = select.select([stream], [], [], 0.1)  # 0.1-second timeout
            if ready:
                try:
                    line = stream.readline()
                    if line:
                        print(line, end="")
                    else:
                        break  # Exit if no more data and process ended
                except ValueError as e:
                    print("Stream closed:", e)
                    break
            if self.port_forward_process.poll() is not None:
                break

    def is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0
        
    def get_jaeger_pod_name(self):
        try:
            result = subprocess.check_output(
                ["kubectl", "get", "pods", "-n", self.namespace,
                 "-l", "app.kubernetes.io/name=jaeger",
                 "-o", "jsonpath={.items[0].metadata.name}"],
                text=True
            )
            return result.strip()
        except subprocess.CalledProcessError as e:
            print("Error getting Jaeger pod name:", e)
            raise

    def start_port_forward(self):
        """Starts kubectl port-forward command to access Jaeger service or pod."""
        for attempt in range(3):
            if self.is_port_in_use(16686):
                print(
                    f"Port 16686 is already in use. Attempt {attempt + 1} of 3. Retrying in 3 seconds..."
                )
                time.sleep(3)
                continue

            # Use pod port-forwarding for astronomy-shop only
            if self.namespace == "astronomy-shop":
                pod_name = self.get_jaeger_pod_name()
                command = f"kubectl port-forward pod/{pod_name} 16686:16686 -n {self.namespace}"
            else:
                command = f"kubectl port-forward svc/jaeger 16686:16686 -n {self.namespace}"

            print("Starting port-forward with command:", command)
            self.port_forward_process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            thread_out = threading.Thread(
                target=self.print_output, args=(self.port_forward_process.stdout,)
            )
            thread_err = threading.Thread(
                target=self.print_output, args=(self.port_forward_process.stderr,)
            )
            thread_out.start()
            thread_err.start()

            time.sleep(3)  # Let port-forward initialize

            if self.port_forward_process.poll() is None:
                print("Port forwarding established successfully.")
                break
            else:
                print("Port forwarding failed. Retrying...")
        else:
            print("Failed to establish port forwarding after 3 attempts.")

        # TODO: modify this command for other microservices
        # command = "kubectl port-forward svc/jaeger 16686:16686 -n hotel-reservation"
        # self.port_forward_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # thread_out = threading.Thread(target=self.print_output, args=(self.port_forward_process.stdout,))
        # thread_err = threading.Thread(target=self.print_output, args=(self.port_forward_process.stderr,))
        # thread_out.start()
        # thread_err.start()
        # # self.output_threads.extend([thread_out, thread_err])
        # time.sleep(3)  # Wait a bit for the port-forward to establish

    def stop_port_forward(self):
        if self.port_forward_process:
            self.stop_event.set()  # Signal threads to exit
            try:
                self.port_forward_process.terminate()
                self.port_forward_process.wait(timeout=5)
            except Exception as e:
                print("Error terminating port-forward process:", e)

            try:
                if self.port_forward_process.stdout:
                    self.port_forward_process.stdout.close()
                if self.port_forward_process.stderr:
                    self.port_forward_process.stderr.close()
            except Exception as e:
                print("Error closing process streams:", e)
            self.port_forward_process = None


    def cleanup(self):
        self.stop_port_forward()
        for thread in self.output_threads:
            thread.join(timeout=5)
            if thread.is_alive():
                print(f"Thread {thread.name} could not be joined and may need to be stopped forcefully.")
        self.output_threads.clear()
        print("Cleanup completed.")

    def get_services(self) -> list:
        """Fetch a list of services from the tracing API."""
        url = f"{self.base_url}/api/services"
        headers = {"Accept": "application/json"} if self.namespace == "astronomy-shop" else {}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            print(f"Failed to get services: {e}")
            return []

    def get_traces(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = None,
    ) -> list:
        """
        Fetch traces for a specific service between start_time and end_time.
        If limit is not specified, all available traces are fetched.
        """
        lookback = int((datetime.now() - start_time).total_seconds())
        url = f"{self.base_url}/api/traces?service={service_name}&lookback={lookback}s"
        if limit is not None:
            url += f"&limit={limit}"

        headers = {"Accept": "application/json"} if self.namespace == "astronomy-shop" else {}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.RequestException as e:
            print(f"Failed to get traces for {service_name}: {e}")
            return []

    def extract_traces(
        self, start_time: datetime, end_time: datetime, limit: int = None
    ) -> list:
        """
        Extract traces for all services between start_time and end_time.
        """
        services = self.get_services()
        print(f"services: {services}")
        all_traces = []
        # Check if services is None - sometimes Jaeger's sampling
        # will lead the number of traces very small or even none
        if services is None:
            print("No services found.")
            return all_traces
        for service in services:
            if service == "jaeger-all-in-one":  # Skip utility service
                continue
            traces = self.get_traces(
                service_name=service,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
            for trace in traces:
                for span in trace["spans"]:
                    span["serviceName"] = trace["processes"][span["processID"]][
                        "serviceName"
                    ]
                all_traces.append(trace)  # Collect the trace with service name included
        self.cleanup()
        print("Cleanup completed.")
        # print(f"all_traces: {all_traces}")
        return all_traces

    def process_traces(self, traces) -> pd.DataFrame:
        """Process raw traces data into a structured DataFrame."""
        trace_id_list = []
        span_id_list = []
        service_name_list = []
        operation_name_list = []
        start_time_list = []
        duration_list = []
        parent_span_list = []
        error_list = []
        response_list = []

        for trace in traces:
            trace_id = trace["traceID"]
            for span in trace["spans"]:
                trace_id_list.append(trace_id)
                span_id_list.append(span["spanID"])
                parent_span = "ROOT"
                if "references" in span:
                    for ref in span["references"]:
                        if ref["refType"] == "CHILD_OF":
                            parent_span = ref["spanID"]
                            break
                parent_span_list.append(parent_span)

                service_name_list.append(
                    span["serviceName"]
                )  # Use the correct service name from the span
                operation_name_list.append(span["operationName"])
                start_time_list.append(span["startTime"])
                duration_list.append(span["duration"])

                has_error = False
                response = "Unknown"
                for tag in span.get("tags", []):
                    if tag["key"] == "error" and tag["value"] == True:
                        has_error = True
                    if tag["key"] == "http.status_code" or tag["key"] == "response_class":
                        response = tag["value"]
                error_list.append(has_error)
                response_list.append(response)

        df = pd.DataFrame(
            {
                "trace_id": trace_id_list,
                "span_id": span_id_list,
                "parent_span": parent_span_list,
                "service_name": service_name_list,
                "operation_name": operation_name_list,
                "start_time": start_time_list,
                "duration": duration_list,
                "has_error": error_list,
                "response": response_list,
            }
        )
        return df

    def save_traces(self, df, path) -> str:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"traces_{int(time.time())}.csv")
        df.to_csv(file_path, index=False)
        self.cleanup() # Stop port-forwarding after traces are exported
        return f"Traces data exported to: {file_path}"


if __name__ == "__main__":
    tracer = TraceAPI(namespace="hotel-reservation")
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=9)  # Example time window
    traces = tracer.extract_traces(start_time, end_time)
    df_traces = tracer.process_traces(traces)
    save_path = root_path / "trace_output"
    tracer.save_traces(df_traces, save_path)
