import subprocess
import os
import logging

REPO = "/home/azureuser/AIOpsLab"

# Configure logging
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for more detailed logs
logger = logging.getLogger(__name__)


def run_command(command, capture_output=False):
    """Runs a shell command and handles errors."""
    try:
        logger.debug(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command, capture_output=capture_output, text=True, check=True
        )
        if capture_output:
            logger.debug(f"Command output: {result.stdout.strip()}")
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Command '{' '.join(command)}' failed with error: {e.stderr.strip() if e.stderr else str(e)}"
        )
        if capture_output:
            return None


def setup_aiopslab():
    try:
        run_command(["terraform", "plan", "-out", "main.tfplan"])
        output = run_command(["terraform", "apply", "main.tfplan"], capture_output=True)
        if output:
            logger.debug(f"Terraform apply output: {output}")
    except Exception as e:
        logger.error(f"Error in setup_aiopslab: {str(e)}")


def destroy_aiopslab():
    pass


def get_terraform_output(output_name):
    """Retrieve Terraform output."""
    try:
        result = run_command(
            ["terraform", "output", "-raw", output_name], capture_output=True
        )
        return result
    except Exception as e:
        logger.error(f"Failed to get Terraform output for {output_name}: {str(e)}")
        return None


def save_private_key(key_data, filename):
    """Save the private key to a file."""
    try:
        with open(filename, "w") as key_file:
            key_file.write(key_data)
        os.chmod(filename, 0o600)
        logger.info(f"Private key saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save private key to {filename}: {str(e)}")


def copy_and_execute_script(username, private_key, public_ip, script):
    """Copy and execute the shell script on the remote VM."""
    remote_path = f"{username}@{public_ip}:/home/{username}"
    try:
        # Copy the shell script to the remote VM
        run_command(
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                private_key,
                script,
                remote_path,
            ]
        )

        # Execute the shell script on the remote VM
        run_command(
            [
                "ssh",
                "-i",
                private_key,
                f"{username}@{public_ip}",
                f"bash /home/{username}/{os.path.basename(script)}",
            ]
        )
    except Exception as e:
        logger.error(f"Failed to copy or execute script on {public_ip}: {str(e)}")


def get_kubeadm_join_remote(username, private_key, public_ip):
    """SSH into the remote machine and generate the kubeadm join command."""
    generate_join_command = [
        "ssh",
        "-i",
        private_key,
        f"{username}@{public_ip}",
        "sudo kubeadm token create --print-join-command",
    ]
    try:
        print(generate_join_command)
        result = run_command(generate_join_command, capture_output=True)
        return result
    except Exception as e:
        logger.error(
            f"Failed to retrieve kubeadm join command from {public_ip}: {str(e)}"
        )
        return None


def run_kubeadm_join_on_worker(worker_username, private_key, worker_ip, join_command):
    """SSH into the worker and run the kubeadm join command."""
    ssh_command = [
        "ssh",
        "-i",
        private_key,
        f"{worker_username}@{worker_ip}",
        f"sudo {join_command} --cri-socket /var/run/cri-dockerd.sock",
    ]
    try:
        run_command(ssh_command)
    except Exception as e:
        logger.error(f"Failed to run kubeadm join on {worker_ip}: {str(e)}")


def add_ssh_key(host, port=22):
    """Runs ssh-keyscan on a given host and appends the key to known_hosts."""
    try:
        # Build the ssh-keyscan command
        keyscan_cmd = ["ssh-keyscan", "-H", "-p", str(port), host]

        # Run the ssh-keyscan command and capture output
        result = run_command(keyscan_cmd, capture_output=True)

        # Append the output (host key) to known_hosts
        with open(os.path.expanduser("~/.ssh/known_hosts"), "a") as known_hosts_file:
            known_hosts_file.write(result)
        logger.info(f"SSH key for {host} added to known_hosts.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to fetch SSH key for {host}: {e}")
    except Exception as ex:
        logger.error(f"An error occurred while adding SSH key for {host}: {ex}")


def deploy_prometheus(username, private_key_file_1, public_ip_1):
    """Deploy Prometheus on the worker node."""
    try:
        run_command(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                private_key_file_1,
                f"{username}@{public_ip_1}",
                f"bash {REPO}/scripts/setup.sh kubeworker1",
            ]
        )
    except Exception as e:
        logger.error(f"Failed to deploy Prometheus: {str(e)}")


def main():
    # Retrieve private keys and public IPs for both VMs
    private_key_1 = get_terraform_output("key_data_1")
    private_key_2 = get_terraform_output("key_data_2")
    public_ip_1 = get_terraform_output("public_ip_address_1")
    public_ip_2 = get_terraform_output("public_ip_address_2")
    username = "azureuser"  # TODO: read from variables file

    if not private_key_1 or not private_key_2 or not public_ip_1 or not public_ip_2:
        logger.error("Failed to retrieve required Terraform outputs.")
        return

    # Save the private keys to files
    private_key_file_1 = "vm_1_private_key.pem"
    private_key_file_2 = "vm_2_private_key.pem"
    save_private_key(private_key_1, private_key_file_1)
    save_private_key(private_key_2, private_key_file_2)

    # Path to the shell script
    kubeadm_shell_script = f"./scripts/kubeadm.sh"
    controller_shell_script = f"./scripts/kube_controller.sh"
    setup_aiopslab_script = f"./scripts/setup_aiopslab.sh"
    prom_worker_setup_script = f"./scripts/prom_on_worker.sh"

    # Install kubeadm on all the VMs
    copy_and_execute_script(
        username, private_key_file_1, public_ip_1, kubeadm_shell_script
    )
    copy_and_execute_script(
        username, private_key_file_2, public_ip_2, kubeadm_shell_script
    )

    # Setup kube controller
    copy_and_execute_script(
        username, private_key_file_1, public_ip_1, controller_shell_script
    )

    # Get join command and run on the worker
    join_command = get_kubeadm_join_remote(username, private_key_file_1, public_ip_1)

    if join_command:
        logger.info(f"Join command retrieved: {join_command}")
        run_kubeadm_join_on_worker(
            username, private_key_file_2, public_ip_2, join_command
        )

    # Setup aiopslab
    copy_and_execute_script(
        username, private_key_file_1, public_ip_1, setup_aiopslab_script
    )

    # Deploy Prometheus on the worker node)
    copy_and_execute_script(
        username, private_key_file_2, public_ip_2, prom_worker_setup_script
    )
    deploy_prometheus(username, private_key_file_1, public_ip_1)

    # print public ip of controller and worker and give ssh command to access it
    logger.info(f"Controller Public IP: {public_ip_1}")
    logger.info(f"Worker Public IP: {public_ip_2}")
    logger.info(
        f"SSH command to access controller: ssh -i {private_key_file_1} {username}@{public_ip_1}"
    )
    logger.info(
        f"SSH command to access worker: ssh -i {private_key_file_2} {username}@{public_ip_2}"
    )


if __name__ == "__main__":
    main()
