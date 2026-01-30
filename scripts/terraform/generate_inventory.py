#!/usr/bin/env python3
"""
Ansible Inventory Generator for AIOpsLab
Generates inventory.yml from Terraform outputs for multi-VM Kubernetes cluster deployment.
"""

import json
import subprocess
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, capture_output=True):
    """Execute a shell command and return output."""
    try:
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=True
        )
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(f"Error: {e.stderr.strip() if e.stderr else str(e)}")
        return None


def get_terraform_outputs():
    """Retrieve all Terraform outputs as JSON."""
    logger.info("Retrieving Terraform outputs...")
    output = run_command(["terraform", "output", "-json"])

    if not output:
        logger.error("Failed to retrieve Terraform outputs")
        logger.error("Make sure you've run 'terraform apply' successfully")
        sys.exit(1)

    try:
        outputs = json.loads(output)
        logger.info("Successfully retrieved Terraform outputs")
        return outputs
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Terraform output JSON: {e}")
        sys.exit(1)


def validate_outputs(outputs):
    """Validate that required outputs exist."""
    required = ['controller', 'workers', 'ssh_config']
    missing = [key for key in required if key not in outputs]

    if missing:
        logger.error(f"Missing required Terraform outputs: {', '.join(missing)}")
        logger.error("Please update your Terraform configuration")
        sys.exit(1)

    logger.info("All required outputs present")


def generate_inventory(outputs):
    """Generate Ansible inventory YAML from Terraform outputs."""
    controller = outputs['controller']['value']
    workers = outputs['workers']['value']
    ssh_config = outputs['ssh_config']['value']

    # Build inventory structure
    # user_home_base: /home for cloud VMs (Linux standard), /users for Emulab
    inventory = {
        'all': {
            'vars': {
                'k8s_user': controller['username'],
                'user_home_base': '/home',  # Cloud VMs use /home, Emulab uses /users
                'ansible_ssh_private_key_file': ssh_config.get('private_key_path', '~/.ssh/id_rsa')
            },
            'children': {
                'control_nodes': {
                    'hosts': {
                        'control_node': {
                            'ansible_host': controller['public_ip'],
                            'ansible_user': controller['username'],
                            'private_ip': controller['private_ip']
                        }
                    }
                },
                'worker_nodes': {
                    'hosts': {}
                }
            }
        }
    }

    # Add all workers dynamically
    for idx, worker in enumerate(workers, start=1):
        worker_name = f"worker_node_{idx}"
        inventory['all']['children']['worker_nodes']['hosts'][worker_name] = {
            'ansible_host': worker['public_ip'],
            'ansible_user': worker['username'],
            'private_ip': worker['private_ip']
        }

    logger.info(f"Generated inventory for 1 controller + {len(workers)} worker(s)")
    return inventory


def write_inventory_yaml(inventory, output_path):
    """Write inventory to YAML file."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Installing...")
        run_command([sys.executable, "-m", "pip", "install", "pyyaml"])
        import yaml

    try:
        with open(output_path, 'w') as f:
            yaml.dump(inventory, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Inventory written to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write inventory file: {e}")
        return False


def print_inventory_summary(inventory):
    """Print a summary of the generated inventory."""
    controller = inventory['all']['children']['control_nodes']['hosts']['control_node']
    workers = inventory['all']['children']['worker_nodes']['hosts']

    print("\n" + "="*60)
    print("ANSIBLE INVENTORY SUMMARY")
    print("="*60)
    print(f"\n📋 Controller Node:")
    print(f"   Name: control_node")
    print(f"   IP:   {controller['ansible_host']}")
    print(f"   User: {controller['ansible_user']}")

    print(f"\n👷 Worker Nodes ({len(workers)}):")
    for name, info in workers.items():
        print(f"   {name}:")
        print(f"      IP:   {info['ansible_host']}")
        print(f"      User: {info['ansible_user']}")

    print("\n" + "="*60)


def main():
    """Main execution function."""
    # Get current directory
    script_dir = Path(__file__).parent
    ansible_dir = script_dir.parent / "ansible"
    inventory_path = ansible_dir / "inventory.yml"

    logger.info("Starting Ansible inventory generation...")
    logger.info(f"Target inventory file: {inventory_path}")

    # Check if we're in the right directory
    if not (script_dir / "main.tf").exists():
        logger.error("main.tf not found. Are you in the terraform directory?")
        sys.exit(1)

    # Create ansible directory if it doesn't exist
    ansible_dir.mkdir(exist_ok=True)

    # Get Terraform outputs
    outputs = get_terraform_outputs()

    # Validate outputs
    validate_outputs(outputs)

    # Generate inventory
    inventory = generate_inventory(outputs)

    # Write inventory file
    if write_inventory_yaml(inventory, inventory_path):
        print_inventory_summary(inventory)
        logger.info("✅ Inventory generation completed successfully!")
        return 0
    else:
        logger.error("❌ Failed to generate inventory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
