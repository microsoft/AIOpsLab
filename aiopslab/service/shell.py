# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Interface to run shell commands in the service cluster."""

import subprocess
import paramiko
import os
from aiopslab.paths import config


class Shell:
    """Interface to run shell commands in the service cluster.

    Note: this can only run a single command and get its output.
    TODO: expand to a stateful shell session interface.
    """

    @staticmethod
    def exec(command: str, input_data=None, cwd=None):
        """Execute a shell command on localhost, via SSH, or inside kind's control-plane container."""
        k8s_host = config.get("k8s_host", "localhost")  # Default to localhost

        if k8s_host == "kind":
            print("[INFO] Running command inside kind-control-plane Docker container.")
            return Shell.docker_exec("kind-control-plane", command)

        elif k8s_host == "localhost":
            print(
                "[WARNING] Running commands on localhost is not recommended. "
                "This may pose safety and security risks when using an AI agent locally. "
                "I hope you know what you're doing!!!"
            )
            return Shell.local_exec(command, input_data, cwd)

        else:
            k8s_user = config.get("k8s_user")
            ssh_key_path = config.get("ssh_key_path", "~/.ssh/id_rsa")
            return Shell.ssh_exec(k8s_host, k8s_user, ssh_key_path, command)

    @staticmethod
    def local_exec(command: str, input_data=None, cwd=None):
        if input_data is not None:
            input_data = input_data.encode("utf-8")

        try:
            out = subprocess.run(
                command,
                input=input_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                cwd=cwd,
            )

            if out.stderr or out.returncode != 0:
                error_message = out.stderr.decode("utf-8")
                print(f"[ERROR] Command execution failed: {error_message}")
                return error_message
            else:
                output_message = out.stdout.decode("utf-8")
                print(output_message)
                return output_message

        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {command}\nError: {str(e)}")

    @staticmethod
    def ssh_exec(host: str, user: str, ssh_key_path: str, command: str):
        ssh_key_path = os.path.expanduser(ssh_key_path)
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh_client.connect(hostname=host, username=user, key_filename=ssh_key_path)

            stdin, stdout, stderr = ssh_client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error_message = stderr.read().decode("utf-8")
                print(f"[ERROR] SSH Command execution failed: {error_message}")
                return error_message
            else:
                output_message = stdout.read().decode("utf-8")
                print(output_message)
                return output_message

        except Exception as e:
            raise RuntimeError(f"Failed to execute command via SSH: {command}\nError: {str(e)}")

        finally:
            ssh_client.close()

    @staticmethod
    def docker_exec(container_name: str, command: str):
        """Execute a command inside a running Docker container."""
        docker_command = f"docker exec {container_name} sh -c '{command}'"

        try:
            out = subprocess.run(
                docker_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            if out.stderr or out.returncode != 0:
                error_message = out.stderr.decode("utf-8")
                print(f"[ERROR] Docker command execution failed: {error_message}")
                return error_message
            else:
                output_message = out.stdout.decode("utf-8")
                print(output_message)
                return output_message

        except Exception as e:
            raise RuntimeError(f"Failed to execute command in Docker container: {container_name}\nError: {str(e)}")