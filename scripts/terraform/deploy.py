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
import sys
import os
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

    def terraform_plan(self, worker_count, vm_size, resource_group, prefix, ssh_key_path):
        """Create Terraform plan."""
        logger.info("Creating Terraform plan...")

        plan_vars = [
            "-out=main.tfplan",
            f"-var=worker_vm_count={worker_count}",
            f"-var=vm_size={vm_size}",
            f"-var=resource_group_name={resource_group}",
            f"-var=prefix={prefix}",
            f"-var=ssh_public_key_path={ssh_key_path}"
        ]

        return self.run_command(["terraform", "plan"] + plan_vars)

    def terraform_plan_only(self, worker_count, vm_size, resource_group, prefix, ssh_key_path):
        """Show Terraform plan without saving (dry-run)."""
        logger.info("Creating Terraform plan (dry-run)...")

        plan_vars = [
            f"-var=worker_vm_count={worker_count}",
            f"-var=vm_size={vm_size}",
            f"-var=resource_group_name={resource_group}",
            f"-var=prefix={prefix}",
            f"-var=ssh_public_key_path={ssh_key_path}"
        ]

        return self.run_command(["terraform", "plan"] + plan_vars)

    def terraform_apply(self):
        """Apply Terraform plan."""
        logger.info("Applying Terraform plan...")
        return self.run_command(["terraform", "apply", "main.tfplan"])

    def terraform_destroy(self, resource_group, prefix, ssh_key_path):
        """Destroy Terraform-managed infrastructure."""
        logger.info("Destroying Terraform infrastructure...")

        # Create destroy plan
        destroy_vars = [
            "-destroy",
            "-out=main.destroy.tfplan",
            f"-var=resource_group_name={resource_group}",
            f"-var=prefix={prefix}",
            f"-var=ssh_public_key_path={ssh_key_path}"
        ]

        logger.info("Creating destroy plan...")
        if not self.run_command(["terraform", "plan"] + destroy_vars):
            return False

        logger.info("Applying destroy plan...")
        return self.run_command(["terraform", "apply", "main.destroy.tfplan"])

    def add_nsg_corpnet_rule(self, resource_group, prefix):
        """Add NSG rule to restrict SSH to Microsoft CorpNet."""
        logger.info("Adding NSG rule to restrict SSH to CorpNetPublic...")

        nsg_name = f"{prefix}-nsg"

        command = [
            "az", "network", "nsg", "rule", "create",
            "-g", resource_group,
            "--nsg-name", nsg_name,
            "--name", "SSH-CorpNet",
            "--priority", "100",
            "--protocol", "TCP",
            "--source-address-prefixes", "CorpNetPublic",
            "--destination-port-ranges", "22",
            "--access", "Allow",
            "--direction", "Inbound"
        ]

        try:
            result = self.run_command(command, check=False)
            if result:
                logger.info("✅ NSG rule 'SSH-CorpNet' added successfully")
                logger.info("   SSH access is now restricted to Microsoft Corporate Network")
                return True
            else:
                logger.warning("⚠️  Failed to add NSG rule. You may need to add it manually.")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to add NSG rule: {e}")
            return False

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

        logger.info("✅ All hosts are SSH-ready")
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
                    logger.warning(f"⚠️  Could not get SSH host key for {host}")

            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️  ssh-keyscan timeout for {host}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to add SSH host key for {host}: {e}")

        logger.info("✅ SSH host keys added")
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
            logger.warning("⚠️  SSH host key checking is DISABLED. This is insecure on untrusted networks.")
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

    def print_access_info(self, outputs):
        """Print SSH access information."""
        controller = outputs['controller']['value']
        workers = outputs['workers']['value']
        ssh_config = outputs['ssh_config']['value']

        print("\n" + "="*70)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("="*70)

        print(f"\n📋 Controller Node:")
        print(f"   Public IP:  {controller['public_ip']}")
        print(f"   Private IP: {controller['private_ip']}")
        print(f"   SSH:        ssh -i {ssh_config.get('private_key_path', '~/.ssh/id_rsa')} {controller['username']}@{controller['public_ip']}")

        print(f"\n👷 Worker Nodes ({len(workers)}):")
        for idx, worker in enumerate(workers, start=1):
            print(f"   Worker {idx}:")
            print(f"      Public IP:  {worker['public_ip']}")
            print(f"      Private IP: {worker['private_ip']}")
            print(f"      SSH:        ssh -i {ssh_config.get('private_key_path', '~/.ssh/id_rsa')} {worker['username']}@{worker['public_ip']}")

        print("\n📝 Next Steps:")
        print(f"   1. SSH into controller: ssh -i {ssh_config.get('private_key_path', '~/.ssh/id_rsa')} {controller['username']}@{controller['public_ip']}")
        print(f"   2. Check cluster: kubectl get nodes")
        print(f"   3. Deploy AIOpsLab: cd ~/AIOpsLab && poetry install")

        print("\n" + "="*70 + "\n")

    def deploy(self, worker_count, vm_size, resource_group, prefix, ssh_key_path, restrict_ssh_corpnet=False, disable_host_key_checking=False):
        """Execute full deployment workflow."""
        logger.info("="*70)
        logger.info("STARTING AIOPSLAB DEPLOYMENT")
        logger.info("="*70)
        logger.info(f"Workers: {worker_count}")
        logger.info(f"VM Size: {vm_size}")
        logger.info(f"Resource Group: {resource_group}")
        logger.info(f"Prefix: {prefix}")
        if restrict_ssh_corpnet:
            logger.info(f"SSH Access: Restricted to CorpNetPublic")
        if disable_host_key_checking:
            logger.warning(f"⚠️  Host key checking: DISABLED (insecure)")

        try:
            # Step 1: Initialize Terraform
            if not self.terraform_init():
                logger.error("Terraform initialization failed")
                return False

            # Step 2: Plan infrastructure
            if not self.terraform_plan(worker_count, vm_size, resource_group, prefix, ssh_key_path):
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

            # Step 4.5: Add CorpNet NSG rule if requested
            if restrict_ssh_corpnet:
                if not self.add_nsg_corpnet_rule(resource_group, prefix):
                    logger.warning("⚠️  NSG rule creation failed, but continuing deployment...")
                    logger.warning("   You may need to add the rule manually later.")

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
                    logger.error("❌ Failed to add SSH host keys automatically")
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

            logger.info("✅ DEPLOYMENT SUCCESSFUL!")
            return True

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Deployment interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False

    def plan(self, worker_count, vm_size, resource_group, prefix, ssh_key_path):
        """Show deployment plan without applying (dry-run)."""
        logger.info("="*70)
        logger.info("AIOPSLAB DEPLOYMENT PLAN (DRY-RUN)")
        logger.info("="*70)
        logger.info(f"Workers: {worker_count}")
        logger.info(f"VM Size: {vm_size}")
        logger.info(f"Resource Group: {resource_group}")
        logger.info(f"Prefix: {prefix}")
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
            if not self.terraform_plan_only(worker_count, vm_size, resource_group, prefix, ssh_key_path):
                logger.error("Terraform planning failed")
                return False

            logger.info("")
            logger.info("="*70)
            logger.info("✅ PLAN COMPLETE!")
            logger.info("="*70)
            logger.info("")
            logger.info("📋 Review the plan above to see what would be created.")
            logger.info("")
            logger.info("To actually deploy, run:")
            logger.info(f"  python3 deploy.py --apply --workers {worker_count} --vm-size {vm_size} \\")
            logger.info(f"      --resource-group {resource_group} --prefix {prefix}")
            logger.info("")
            logger.info("⚠️  Note: This will create billable Azure resources.")
            logger.info("")
            return True

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Plan interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Plan failed: {e}")
            return False

    def destroy(self, resource_group, prefix, ssh_key_path):
        """Destroy deployed infrastructure."""
        logger.info("="*70)
        logger.info("DESTROYING AIOPSLAB INFRASTRUCTURE")
        logger.info("="*70)

        try:
            # Confirm destruction
            confirm = input("⚠️  This will destroy all resources. Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                logger.info("Destruction cancelled")
                return False

            # Destroy infrastructure
            if self.terraform_destroy(resource_group, prefix, ssh_key_path):
                logger.info("✅ Infrastructure destroyed successfully")
                return True
            else:
                logger.error("❌ Destruction failed")
                return False

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Destruction interrupted by user")
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

  Deploy with SSH restricted to Microsoft CorpNet:
    python deploy.py --apply --workers 2 --vm-size Standard_B2s --restrict-ssh-corpnet

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
        '--restrict-ssh-corpnet',
        action='store_true',
        help='Restrict SSH access to Microsoft CorpNetPublic (adds NSG rule with priority 100)'
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
            ssh_key_path=ssh_key_path
        )
    elif args.apply:
        success = deployer.deploy(
            worker_count=args.workers,
            vm_size=args.vm_size,
            resource_group=args.resource_group,
            prefix=args.prefix,
            ssh_key_path=ssh_key_path,
            restrict_ssh_corpnet=args.restrict_ssh_corpnet,
            disable_host_key_checking=args.disable_host_key_checking
        )
    elif args.destroy:
        success = deployer.destroy(
            resource_group=args.resource_group,
            prefix=args.prefix,
            ssh_key_path=ssh_key_path
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
