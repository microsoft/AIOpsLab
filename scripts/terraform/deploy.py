#!/usr/bin/env python3
"""
AIOpsLab Automated Deployment Script
Provisions Azure VMs with Terraform and sets up Kubernetes cluster with Ansible.

Usage:
    python deploy.py --plan --workers 3 --vm-size Standard_D4s_v3
    python deploy.py --apply --workers 3 --vm-size Standard_D4s_v3
    python deploy.py --destroy
    python deploy.py --help
"""

import subprocess
import shutil
import sys
import os
import re
import time
import argparse
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIOpsLabDeployer:
    """Main deployment orchestrator for AIOpsLab."""

    def __init__(self, terraform_dir=None, ansible_dir=None):
        """Initialize deployer with directory paths."""
        self.script_dir = Path(__file__).parent
        self.terraform_dir = Path(terraform_dir) if terraform_dir else self.script_dir
        self.ansible_dir = Path(ansible_dir) if ansible_dir else self.script_dir.parent / "ansible"
        self.inventory_path = self.ansible_dir / "inventory.yml"

    def run_command(self, command, capture_output=False, cwd=None, check=True):
        """Execute a shell command."""
        try:
            logger.debug(f"Running: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                check=check,
                cwd=cwd or self.terraform_dir
            )
            if capture_output:
                return result.stdout.strip()
            return result.returncode == 0
        except FileNotFoundError:
            logger.error(f"Command not found: {command[0]}")
            logger.error(f"Please ensure '{command[0]}' is installed and in your PATH")
            if check:
                raise
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(command)}")
            if e.stderr:
                logger.error(f"Error: {e.stderr.strip()}")
            if check:
                raise
            return False

    def terraform_init(self):
        """Initialize Terraform."""
        logger.info("Initializing Terraform...")
        return self.run_command(["terraform", "init"])

    def terraform_plan(self, worker_count, vm_size, resource_group, prefix, ssh_key_path, allowed_ips="*"):
        """Create Terraform plan."""
        logger.info("Creating Terraform plan...")

        plan_vars = [
            "-out=main.tfplan",
            f"-var=worker_vm_count={worker_count}",
            f"-var=vm_size={vm_size}",
            f"-var=resource_group_name={resource_group}",
            f"-var=prefix={prefix}",
            f"-var=ssh_public_key_path={ssh_key_path}",
            f"-var=nsg_allowed_source={allowed_ips}"
        ]

        return self.run_command(["terraform", "plan"] + plan_vars)

    def terraform_plan_only(self, worker_count, vm_size, resource_group, prefix, ssh_key_path, allowed_ips="*"):
        """Show Terraform plan without saving (dry-run)."""
        logger.info("Creating Terraform plan (dry-run)...")

        plan_vars = [
            f"-var=worker_vm_count={worker_count}",
            f"-var=vm_size={vm_size}",
            f"-var=resource_group_name={resource_group}",
            f"-var=prefix={prefix}",
            f"-var=ssh_public_key_path={ssh_key_path}",
            f"-var=nsg_allowed_source={allowed_ips}"
        ]

        return self.run_command(["terraform", "plan"] + plan_vars)

    def terraform_apply(self):
        """Apply Terraform plan."""
        logger.info("Applying Terraform plan...")
        return self.run_command(["terraform", "apply", "main.tfplan"])

    def terraform_destroy(self, resource_group, prefix, ssh_key_path, allowed_ips="*"):
        """Destroy Terraform-managed infrastructure."""
        logger.info("Destroying Terraform infrastructure...")

        # Create destroy plan
        destroy_vars = [
            "-destroy",
            "-out=main.destroy.tfplan",
            f"-var=resource_group_name={resource_group}",
            f"-var=prefix={prefix}",
            f"-var=ssh_public_key_path={ssh_key_path}",
            f"-var=nsg_allowed_source={allowed_ips}"
        ]

        logger.info("Creating destroy plan...")
        if not self.run_command(["terraform", "plan"] + destroy_vars):
            return False

        logger.info("Applying destroy plan...")
        return self.run_command(["terraform", "apply", "main.destroy.tfplan"])

    def get_terraform_outputs(self):
        """Retrieve Terraform outputs as JSON."""
        logger.info("Retrieving Terraform outputs...")
        output = self.run_command(
            ["terraform", "output", "-json"],
            capture_output=True
        )

        if not output:
            logger.error("Failed to retrieve Terraform outputs")
            return None

        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Terraform outputs: {e}")
            return None

    def generate_ansible_inventory(self):
        """Generate Ansible inventory from Terraform outputs."""
        logger.info("Generating Ansible inventory...")

        inventory_script = self.terraform_dir / "generate_inventory.py"
        if not inventory_script.exists():
            logger.error(f"Inventory generator not found: {inventory_script}")
            return False

        return self.run_command([sys.executable, str(inventory_script)])

    def wait_for_ssh(self, host, port=22, timeout=300, interval=10):
        """Wait for SSH to become available on a host."""
        logger.info(f"Waiting for SSH on {host}...")

        import socket
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()

                if result == 0:
                    logger.info(f"SSH available on {host}")
                    return True

            except socket.error:
                pass

            time.sleep(interval)

        logger.error(f"SSH timeout on {host} after {timeout}s")
        return False

    def wait_for_all_hosts(self, outputs):
        """Wait for SSH on all hosts."""
        controller = outputs['controller']['value']
        workers = outputs['workers']['value']

        logger.info("Waiting for SSH on all hosts...")

        # Wait for controller
        if not self.wait_for_ssh(controller['public_ip']):
            logger.error("Controller SSH not available")
            return False

        # Wait for all workers
        for idx, worker in enumerate(workers, start=1):
            if not self.wait_for_ssh(worker['public_ip']):
                logger.error(f"Worker {idx} SSH not available")
                return False

        logger.info("All hosts are SSH-ready")
        return True

    def add_ssh_host_keys(self, outputs):
        """Add SSH host keys to known_hosts to avoid verification prompts."""
        controller = outputs['controller']['value']
        workers = outputs['workers']['value']

        logger.info("Adding SSH host keys to known_hosts...")

        all_hosts = [controller['public_ip']] + [w['public_ip'] for w in workers]

        for host in all_hosts:
            try:
                # Run ssh-keyscan to get host keys
                result = subprocess.run(
                    ['ssh-keyscan', '-H', host],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0 and result.stdout:
                    # Append to known_hosts
                    known_hosts_path = Path.home() / '.ssh' / 'known_hosts'
                    known_hosts_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(known_hosts_path, 'a') as f:
                        f.write(result.stdout)

                    logger.debug(f"Added SSH host key for {host}")
                else:
                    logger.warning(f"Could not get SSH host key for {host}")

            except subprocess.TimeoutExpired:
                logger.warning(f"ssh-keyscan timeout for {host}")
            except Exception as e:
                logger.warning(f"Failed to add SSH host key for {host}: {e}")

        logger.info("SSH host keys added")
        return True

    def run_ansible_playbook(self, playbook_name, extra_args=None, disable_host_key_checking=False):
        """Run an Ansible playbook."""
        playbook_path = self.ansible_dir / playbook_name

        if not playbook_path.exists():
            logger.error(f"Playbook not found: {playbook_path}")
            return False

        if not self.inventory_path.exists():
            logger.error(f"Inventory not found: {self.inventory_path}")
            return False

        logger.info(f"Running Ansible playbook: {playbook_name}")

        command = [
            "ansible-playbook",
            "-i", str(self.inventory_path),
            str(playbook_path)
        ]

        if extra_args:
            command.extend(extra_args)

        # Optionally disable host key checking (not recommended for production)
        env = os.environ.copy()
        if disable_host_key_checking:
            logger.warning("SSH host key checking is DISABLED. This is insecure on untrusted networks.")
            env['ANSIBLE_HOST_KEY_CHECKING'] = 'False'

        try:
            result = subprocess.run(
                command,
                cwd=self.ansible_dir,
                env=env,
                check=True
            )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Ansible playbook failed with exit code {e.returncode}")
            return False

    def _install_tool(self, name, install_cmd):
        """Try to install a tool via a shell command. Returns True if tool is on PATH after."""
        logger.info(f"Installing {name}...")
        try:
            subprocess.run(install_cmd, shell=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Install failed for {name}: {e}")
            return False
        # Refresh PATH check (poetry installs to ~/.local/bin)
        if name == "poetry":
            local_bin = str(Path.home() / ".local" / "bin")
            if local_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"
        return shutil.which(name) is not None

    def _find_python311_plus(self):
        """Find a Python >= 3.11 binary. Returns (command_name, version_string) or (None, None)."""
        for candidate in ["python3.11", "python3.12", "python3.13", "python3"]:
            path = shutil.which(candidate)
            if not path:
                continue
            try:
                ver = subprocess.run(
                    [path, "--version"], capture_output=True, text=True, check=True
                ).stdout.strip()
                minor = int(ver.split(".")[1])
                if minor >= 11:
                    return candidate, ver
            except Exception:
                continue
        return None, None

    def setup_aiopslab_mode_b(self, outputs, ssh_key_path):
        """Configure AIOpsLab for Mode B (laptop with remote kubectl)."""
        controller = outputs['controller']['value']
        controller_ip = controller['public_ip']
        admin_username = controller['username']
        repo_root = self.script_dir.parent.parent
        kubeconfig_path = Path.home() / ".kube" / "config"

        # Derive private key path (strip .pub if needed)
        ssh_private_key = ssh_key_path
        if ssh_private_key.endswith('.pub'):
            ssh_private_key = ssh_private_key[:-4]

        # Track results: (step_name, status, detail)
        results = []

        # --- kubectl ---
        logger.info("Checking kubectl...")
        if not shutil.which("kubectl"):
            installed = self._install_tool("kubectl",
                'curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"'
                ' && sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl'
                ' && rm -f kubectl')
            if installed:
                results.append(("kubectl", "INSTALLED", shutil.which("kubectl")))
            else:
                results.append(("kubectl", "FAILED", "Auto-install failed, install manually"))
        else:
            results.append(("kubectl", "OK", shutil.which("kubectl")))

        # --- helm ---
        logger.info("Checking helm...")
        if not shutil.which("helm"):
            installed = self._install_tool("helm",
                'curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash')
            if installed:
                results.append(("helm", "INSTALLED", shutil.which("helm")))
            else:
                results.append(("helm", "FAILED", "Auto-install failed, install manually"))
        else:
            results.append(("helm", "OK", shutil.which("helm")))

        # --- kubeconfig ---
        logger.info("Checking kubeconfig...")
        kubeconfig_ok = kubeconfig_path.exists()
        if kubeconfig_ok:
            results.append(("kubeconfig", "OK", str(kubeconfig_path)))
        else:
            results.append(("kubeconfig", "MISSING", "Ansible should have copied it to ~/.kube/config"))

        # --- cluster access ---
        has_kubectl = shutil.which("kubectl") is not None
        if has_kubectl and kubeconfig_ok:
            logger.info("Verifying kubectl connectivity...")
            try:
                self.run_command(["kubectl", "get", "nodes"], check=True)
                results.append(("cluster access", "OK", "kubectl get nodes succeeded"))
            except Exception:
                results.append(("cluster access", "FAILED", "Check NSG rules and kubeconfig server IP"))
        else:
            results.append(("cluster access", "SKIPPED",
                            "No kubectl" if not has_kubectl else "No kubeconfig"))

        # --- config.yml ---
        logger.info("Generating aiopslab/config.yml...")
        config_example = repo_root / "aiopslab" / "config.yml.example"
        config_dest = repo_root / "aiopslab" / "config.yml"

        if not config_example.exists() and not config_dest.exists():
            results.append(("config.yml", "FAILED", "config.yml.example not found"))
        elif config_dest.exists():
            content = config_dest.read_text()
            content = re.sub(r'k8s_host:.*', f'k8s_host: {controller_ip}', content)
            content = re.sub(r'k8s_user:.*', f'k8s_user: {admin_username}', content)
            content = re.sub(r'ssh_key_path:.*', f'ssh_key_path: {ssh_private_key}', content)
            config_dest.write_text(content)
            results.append(("config.yml", "UPDATED", f"k8s_host={controller_ip}"))
        else:
            content = config_example.read_text()
            content = content.replace("k8s_host: control_node_hostname", f"k8s_host: {controller_ip}")
            content = content.replace("k8s_user: your_username", f"k8s_user: {admin_username}")
            content = content.replace("ssh_key_path: ~/.ssh/id_rsa", f"ssh_key_path: {ssh_private_key}")
            config_dest.write_text(content)
            results.append(("config.yml", "OK", f"Generated with k8s_host={controller_ip}"))

        # --- git submodules ---
        logger.info("Checking git submodules...")
        submodules_dir = repo_root / "aiopslab-applications"
        has_content = (submodules_dir.exists()
                       and any(f for f in submodules_dir.iterdir() if f.name != '.git'))
        if has_content:
            results.append(("git submodules", "OK", "aiopslab-applications present"))
        else:
            try:
                self.run_command(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=str(repo_root), check=True)
                results.append(("git submodules", "OK", "Initialized successfully"))
            except Exception:
                git_path = repo_root / ".git"
                is_worktree = git_path.is_file()
                if is_worktree:
                    results.append(("git submodules", "FAILED",
                                    "Worktree detected -- run from Git Bash, not WSL"))
                else:
                    results.append(("git submodules", "FAILED",
                                    "Run: git submodule update --init --recursive"))

        # --- poetry ---
        logger.info("Checking poetry...")
        if not shutil.which("poetry"):
            installed = self._install_tool("poetry",
                'curl -sSL https://install.python-poetry.org | python3 -')
            if installed:
                results.append(("poetry", "INSTALLED", shutil.which("poetry")))
            else:
                results.append(("poetry", "FAILED", "Auto-install failed, install manually"))
        else:
            results.append(("poetry", "OK", shutil.which("poetry")))

        # --- python 3.11+ ---
        logger.info("Checking Python version...")
        python_cmd, python_ver = self._find_python311_plus()
        if python_cmd:
            results.append(("python 3.11+", "OK", python_ver))
        else:
            results.append(("python 3.11+", "MISSING", "Install python3.11 or newer"))

        # --- poetry env + install ---
        has_poetry = shutil.which("poetry") is not None
        if has_poetry and python_cmd:
            logger.info("Running poetry env use + poetry install...")
            try:
                self.run_command(
                    ["poetry", "env", "use", python_cmd],
                    cwd=str(repo_root), check=True)
                self.run_command(
                    ["poetry", "install"],
                    cwd=str(repo_root), check=True)
                results.append(("poetry install", "OK", "Dependencies installed"))
            except Exception:
                results.append(("poetry install", "FAILED",
                                f"Run: poetry env use {python_cmd} && poetry install"))
        elif has_poetry:
            results.append(("poetry install", "SKIPPED", "No compatible Python found"))
        else:
            results.append(("poetry install", "SKIPPED", "Poetry not available"))

        # --- Summary table ---
        ok_statuses = {"OK", "INSTALLED", "UPDATED", "SKIPPED"}
        print("\n" + "="*70)
        print("MODE B SETUP SUMMARY")
        print("="*70)
        print(f"  {'Step':<20} {'Status':<16} {'Detail'}")
        print(f"  {'-'*18:<20} {'-'*14:<16} {'-'*30}")
        needs_action = False
        for name, status, detail in results:
            print(f"  {name:<20} {status:<16} {detail}")
            if status not in ok_statuses:
                needs_action = True

        if not needs_action:
            print(f"\nAll steps completed. To start:")
            print(f"  cd {repo_root}")
            print(f"  eval $(poetry env activate)")
            print(f"  python3 cli.py")
        else:
            print(f"\nSome steps need manual action. See details above.")
            print(f"After resolving, start with:")
            print(f"  cd {repo_root}")
            print(f"  poetry env use python3.11 && poetry install")
            print(f"  eval $(poetry env activate)")
            print(f"  python3 cli.py")

        print("="*70 + "\n")

    def setup_aiopslab_mode_a(self, outputs):
        """Configure AIOpsLab for Mode A (on controller VM). TODO."""
        logger.warning("Mode A setup not yet implemented.")
        logger.info("To set up manually, SSH into the controller and follow the docs.")

    def print_access_info(self, outputs):
        """Print SSH access information."""
        controller = outputs['controller']['value']
        workers = outputs['workers']['value']
        ssh_config = outputs['ssh_config']['value']

        print("\n" + "="*70)
        print("DEPLOYMENT COMPLETE!")
        print("="*70)

        print(f"\nController Node:")
        print(f"   Public IP:  {controller['public_ip']}")
        print(f"   Private IP: {controller['private_ip']}")
        print(f"   SSH:        ssh -i {ssh_config.get('private_key_path', '~/.ssh/id_rsa')} {controller['username']}@{controller['public_ip']}")

        print(f"\nWorker Nodes ({len(workers)}):")
        for idx, worker in enumerate(workers, start=1):
            print(f"   Worker {idx}:")
            print(f"      Public IP:  {worker['public_ip']}")
            print(f"      Private IP: {worker['private_ip']}")
            print(f"      SSH:        ssh -i {ssh_config.get('private_key_path', '~/.ssh/id_rsa')} {worker['username']}@{worker['public_ip']}")

        print("\n" + "="*70 + "\n")

    def deploy(self, worker_count, vm_size, resource_group, prefix, ssh_key_path, allowed_ips="*", disable_host_key_checking=False, mode="B"):
        """Execute full deployment workflow."""
        logger.info("="*70)
        logger.info("STARTING AIOPSLAB DEPLOYMENT")
        logger.info("="*70)
        logger.info(f"Workers: {worker_count}")
        logger.info(f"VM Size: {vm_size}")
        logger.info(f"Resource Group: {resource_group}")
        logger.info(f"Prefix: {prefix}")
        logger.info(f"NSG Allowed Source: {allowed_ips}")
        logger.info(f"Mode: {mode} ({'AIOpsLab on controller' if mode == 'A' else 'AIOpsLab on laptop'})")
        if disable_host_key_checking:
            logger.warning("Host key checking: DISABLED (insecure)")

        try:
            # Step 1: Initialize Terraform
            if not self.terraform_init():
                logger.error("Terraform initialization failed")
                return False

            # Step 2: Plan infrastructure
            if not self.terraform_plan(worker_count, vm_size, resource_group, prefix, ssh_key_path, allowed_ips):
                logger.error("Terraform planning failed")
                return False

            # Step 3: Apply infrastructure
            if not self.terraform_apply():
                logger.error("Terraform apply failed")
                return False

            # Step 4: Get outputs
            outputs = self.get_terraform_outputs()
            if not outputs:
                logger.error("Failed to retrieve Terraform outputs")
                return False

            # Step 5: Generate Ansible inventory
            if not self.generate_ansible_inventory():
                logger.error("Inventory generation failed")
                return False

            # Step 6: Wait for SSH
            if not self.wait_for_all_hosts(outputs):
                logger.error("SSH connectivity check failed")
                return False

            # Step 6.5: Add SSH host keys to known_hosts
            if not disable_host_key_checking:
                keys_added = self.add_ssh_host_keys(outputs)
                if not keys_added:
                    logger.error("Failed to add SSH host keys automatically")
                    logger.error("This is required for secure Ansible execution.")
                    logger.error("Possible causes:")
                    logger.error("  - Firewall blocking SSH port 22")
                    logger.error("  - Network connectivity issues")
                    logger.error("  - SSH service not fully started on VMs")
                    logger.error("\nTo bypass (less secure), use: --disable-host-key-checking")
                    return False

            # Step 7: Run Ansible - setup common
            logger.info("Setting up common dependencies on all nodes...")
            if not self.run_ansible_playbook("setup_common.yml", disable_host_key_checking=disable_host_key_checking):
                logger.error("Ansible setup_common.yml failed")
                return False

            # Step 8: Run Ansible - setup cluster
            logger.info("Setting up Kubernetes cluster...")
            if not self.run_ansible_playbook("remote_setup_controller_worker.yml", disable_host_key_checking=disable_host_key_checking):
                logger.error("Ansible remote_setup_controller_worker.yml failed")
                return False

            # Step 9: Print access info
            self.print_access_info(outputs)

            # Step 10: Set up AIOpsLab
            if mode == 'B':
                self.setup_aiopslab_mode_b(outputs, ssh_key_path)
            elif mode == 'A':
                self.setup_aiopslab_mode_a(outputs)

            logger.info("DEPLOYMENT SUCCESSFUL!")
            return True

        except KeyboardInterrupt:
            logger.warning("\nDeployment interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False

    def plan(self, worker_count, vm_size, resource_group, prefix, ssh_key_path, allowed_ips="*", mode="B"):
        """Show deployment plan without applying (dry-run)."""
        logger.info("="*70)
        logger.info("AIOPSLAB DEPLOYMENT PLAN (DRY-RUN)")
        logger.info("="*70)
        logger.info(f"Workers: {worker_count}")
        logger.info(f"VM Size: {vm_size}")
        logger.info(f"Resource Group: {resource_group}")
        logger.info(f"Prefix: {prefix}")
        logger.info(f"NSG Allowed Source: {allowed_ips}")
        logger.info(f"Mode: {mode} ({'AIOpsLab on controller' if mode == 'A' else 'AIOpsLab on laptop'})")
        logger.info("")
        logger.info("This will show what resources would be created WITHOUT actually creating them.")
        logger.info("="*70)
        logger.info("")

        try:
            # Step 1: Initialize Terraform
            if not self.terraform_init():
                logger.error("Terraform initialization failed")
                return False

            # Step 2: Show plan only (no apply)
            if not self.terraform_plan_only(worker_count, vm_size, resource_group, prefix, ssh_key_path, allowed_ips):
                logger.error("Terraform planning failed")
                return False

            logger.info("")
            logger.info("="*70)
            logger.info("PLAN COMPLETE!")
            logger.info("="*70)
            logger.info("")
            logger.info("Review the plan above to see what would be created.")
            logger.info("")
            logger.info("To actually deploy, run:")
            apply_args = [a if a != "--plan" else "--apply" for a in sys.argv]
            logger.info(f"  {' '.join(apply_args)}")
            logger.info("")
            logger.info("Note: This will create billable Azure resources.")
            logger.info("")
            return True

        except KeyboardInterrupt:
            logger.warning("\nPlan interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Plan failed: {e}")
            return False

    def destroy(self, resource_group, prefix, ssh_key_path, allowed_ips="*"):
        """Destroy deployed infrastructure."""
        logger.info("="*70)
        logger.info("DESTROYING AIOPSLAB INFRASTRUCTURE")
        logger.info("="*70)

        try:
            # Confirm destruction
            confirm = input("This will destroy all resources. Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                logger.info("Destruction cancelled")
                return False

            # Destroy infrastructure
            if self.terraform_destroy(resource_group, prefix, ssh_key_path, allowed_ips):
                logger.info("Infrastructure destroyed successfully")
                return True
            else:
                logger.error("Destruction failed")
                return False

        except KeyboardInterrupt:
            logger.warning("\nDestruction interrupted by user")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AIOpsLab Automated Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Show deployment plan (dry-run):
    python deploy.py --plan --workers 3 --vm-size Standard_D4s_v3

  Deploy with 3 workers:
    python deploy.py --apply --workers 3 --vm-size Standard_D4s_v3

  Deploy with SSH restricted to a specific CIDR:
    python deploy.py --apply --workers 2 --allowed-ips 203.0.113.0/24

  Deploy with SSH restricted to an Azure service tag:
    python deploy.py --apply --workers 2 --allowed-ips CorpNetPublic

  Destroy infrastructure:
    python deploy.py --destroy

  Deploy with custom settings:
    python deploy.py --apply --workers 5 --vm-size Standard_D8s_v3 \\
        --resource-group my-rg --prefix myaiops --ssh-key ~/.ssh/id_rsa.pub
        """
    )

    parser.add_argument(
        '--plan',
        action='store_true',
        help='Show deployment plan without applying (dry-run)'
    )

    parser.add_argument(
        '--apply',
        action='store_true',
        help='Deploy infrastructure and setup cluster'
    )

    parser.add_argument(
        '--destroy',
        action='store_true',
        help='Destroy all infrastructure'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of worker nodes (default: 2)'
    )

    parser.add_argument(
        '--vm-size',
        default='Standard_B2s',
        help='Azure VM size (default: Standard_B2s)'
    )

    parser.add_argument(
        '--resource-group',
        default='aiopslab-rg',
        help='Azure resource group name (default: aiopslab-rg)'
    )

    parser.add_argument(
        '--prefix',
        default='aiopslab',
        help='Resource name prefix (default: aiopslab)'
    )

    parser.add_argument(
        '--ssh-key',
        default='~/.ssh/id_rsa.pub',
        help='Path to SSH public key (default: ~/.ssh/id_rsa.pub)'
    )

    parser.add_argument(
        '--allowed-ips',
        default='*',
        help='Source address for NSG rules (SSH + K8s API). Use \'*\' for open access (default), a CIDR like \'203.0.113.0/24\', or an Azure service tag like \'CorpNetPublic\''
    )

    parser.add_argument(
        '--mode',
        choices=['A', 'B'],
        default='B',
        help='Deployment mode. A: AIOpsLab runs on controller VM. B: AIOpsLab runs on laptop with remote kubectl (default: B)'
    )

    parser.add_argument(
        '--disable-host-key-checking',
        action='store_true',
        help='Disable SSH host key verification (INSECURE - rarely needed since keys are added automatically. Only use if ssh-keyscan fails or in CI/CD with ephemeral runners)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Validate arguments
    if not args.plan and not args.apply and not args.destroy:
        parser.print_help()
        sys.exit(1)

    # Check for conflicting options
    action_count = sum([args.plan, args.apply, args.destroy])
    if action_count > 1:
        logger.error("Cannot use --plan, --apply, and --destroy together. Choose one.")
        sys.exit(1)

    # Expand SSH key path
    ssh_key_path = os.path.expanduser(args.ssh_key)

    # Create deployer
    deployer = AIOpsLabDeployer()

    # Execute action
    if args.plan:
        success = deployer.plan(
            worker_count=args.workers,
            vm_size=args.vm_size,
            resource_group=args.resource_group,
            prefix=args.prefix,
            ssh_key_path=ssh_key_path,
            allowed_ips=args.allowed_ips,
            mode=args.mode
        )
    elif args.apply:
        success = deployer.deploy(
            worker_count=args.workers,
            vm_size=args.vm_size,
            resource_group=args.resource_group,
            prefix=args.prefix,
            ssh_key_path=ssh_key_path,
            allowed_ips=args.allowed_ips,
            disable_host_key_checking=args.disable_host_key_checking,
            mode=args.mode
        )
    elif args.destroy:
        success = deployer.destroy(
            resource_group=args.resource_group,
            prefix=args.prefix,
            ssh_key_path=ssh_key_path,
            allowed_ips=args.allowed_ips
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
