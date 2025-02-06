# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Inject faults at the OS layer."""

import json
import yaml
import subprocess

from aiopslab.service.kubectl import KubeCtl
from aiopslab.generators.fault.base import FaultInjector
from aiopslab.paths import BASE_DIR
from aiopslab.generators.fault.helpers import (
    get_pids_by_name,
    sn_svc_process_names,
    hr_svc_process_names,
    hr_mongod_process_names,
)


class OSFaultInjector(FaultInjector):
    def __init__(self):
        pass

    # O.1: Kernel issue via the BPF filter
    def kernel_bug(self):
        return NotImplementedError

    # O.2: Simulate a disk woreout failure
    def inject_disk_woreout(self):
        pids = []
        proc_names = hr_mongod_process_names  # if it is SocialNetwork
        for term in proc_names:
            term_pids = get_pids_by_name(term)
            print(f"Found PIDs for term '{term}': {term_pids}")
            pids.extend(term_pids)

        print(f"Injecting kernel fault into processes: {pids}")

        target_syscall = "write"  # syscall for disk I/O
        error_code = -5  # EIO (Input/output error)

        if not pids:
            print("No processes found to inject faults.")
            return
        try:
            # Run err_inject with the target syscall, error code, and PIDs
            # ./err_inject <target_syscall> <error_code> <pid1> [<pid2> ... <pidN>]
            command = [
                "sudo",
                str(BASE_DIR / "generators/fault/bpf_injector/err_inject"),
                target_syscall,
                str(error_code),
            ] + [str(pid) for pid in pids]
            # print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to inject fault: {e}")

        # for pid in pids:
        #     try:
        #         print(f"Stopping process {pid}")
        #         subprocess.run(["sudo", "kill", "-9", str(pid)], check=True)
        #     except subprocess.CalledProcessError as e:
        #         print(f"Failed to stop process {pid}: {e}")

    def recover_disk_woreout(self):
        bpf_folder_path = "/sys/fs/bpf/err_inject"
        try:
            command = ["sudo", "rm", "-rf", bpf_folder_path]
            print(f"Removing folder: {bpf_folder_path}")
            subprocess.run(command, check=True)
            print(f"Successfully removed {bpf_folder_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove folder {bpf_folder_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def main():
    injector = OSFaultInjector()
    injector.inject_disk_woreout()


if __name__ == "__main__":
    main()
