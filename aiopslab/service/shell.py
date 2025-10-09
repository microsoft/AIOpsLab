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
    def exec(command: str, input_data=None, cwd=None, timeout=30):
        """Execute a shell command on localhost, via SSH, or inside kind's control-plane container."""
        k8s_host = config.get("k8s_host", "localhost")  # Default to localhost
        
        if k8s_host == "kind":
            return Shell.docker_exec("kind-control-plane", command, timeout=timeout)

        elif k8s_host == "localhost":
            # print(
            #     "[WARNING] Running commands on localhost is not recommended. "
            #     "This may pose safety and security risks when using an AI agent locally. "
            #     "I hope you know what you're doing!!!"
            # )
            return Shell.local_exec(command, input_data, cwd, timeout=timeout)

        else:
            k8s_user = config.get("k8s_user")
            ssh_key_path = config.get("ssh_key_path", "~/.ssh/id_rsa")
            return Shell.ssh_exec(k8s_host, k8s_user, ssh_key_path, command, timeout=timeout)

    @staticmethod
    def local_exec(command: str, input_data=None, cwd=None, timeout=30):
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
                timeout=timeout,
            )

            if out.returncode != 0:
                error_message = out.stderr.decode("utf-8")
                error_message = f"[ERROR] Command execution failed: {error_message}"
                return error_message
            else:
                output_message = out.stdout.decode("utf-8") + out.stderr.decode("utf-8")
                return output_message

        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {command}\nError: {str(e)}")

    @staticmethod
    def ssh_exec(host: str, user: str, ssh_key_path: str, command: str, timeout=30):
        ssh_key_path = os.path.expanduser(ssh_key_path)
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # Add timeout to SSH connection
            ssh_client.connect(hostname=host, username=user, key_filename=ssh_key_path, timeout=timeout)

            # Execute command with timeout
            stdin, stdout, stderr = ssh_client.exec_command(command, timeout=timeout)
            
            # Set timeout on channel to prevent hanging on recv_exit_status
            stdout.channel.settimeout(timeout)
            stderr.channel.settimeout(timeout)
            
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error_message = stderr.read().decode("utf-8")
                return error_message
            else:
                output_message = stdout.read().decode("utf-8")
                return output_message

        except Exception as e:
            raise RuntimeError(f"Failed to execute command via SSH: {command}\nError: {str(e)}")

        finally:
            ssh_client.close()

    @staticmethod
    def docker_exec(container_name: str, command: str, timeout=30):
        """Execute a command inside a running Docker container."""
        escaped_command = command.replace('"', '\\"')
        
        docker_command = f'docker exec {container_name} sh -c "{escaped_command}"'

        try:
            out = subprocess.run(
                docker_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                timeout=timeout
            )

            if out.returncode != 0:
                error_message = out.stderr.decode("utf-8")
                return f"[ERROR] Docker command execution failed: {error_message}"
            else:
                output_message = out.stdout.decode("utf-8")
                return output_message

        except Exception as e:
            raise RuntimeError(f"Failed to execute command in Docker container: {container_name}\nError: {str(e)}")
