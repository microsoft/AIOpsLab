#!/usr/bin/env python3
"""
Unified deployment script for AIOpsLab that integrates Terraform provisioning
with Ansible configuration management to reduce manual steps.
"""

import subprocess
import os
import logging
import json
import yaml
import time
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIOpsLabDeployment:
    """Unified deployment manager for AIOpsLab infrastructure and configuration."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.terraform_dir = Path(__file__).parent
        self.ansible_dir = Path(__file__).parent.parent / "ansible"
        self.outputs = {}
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load deployment configuration."""
        default_config = {
            "resource_group_name": "",
            "resource_name_prefix": "",
            "resource_location": "westus2",
            "create_resource_group": False,
            "subscription_id": "",
            "username": "azureuser",
            "ansible_inventory_file": "inventory_generated.yml",
            "ssh_timeout": 300,
            "provisioning_timeout": 1200
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            default_config.update(config)
        
        return default_config
    
    def run_command(self, command: list, capture_output: bool = False, 
                   cwd: Optional[str] = None, timeout: Optional[int] = None) -> Optional[str]:
        """Execute a shell command with proper error handling."""
        try:
            logger.debug(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command, 
                capture_output=capture_output, 
                text=True, 
                check=True,
                cwd=cwd or self.terraform_dir,
                timeout=timeout
            )
            if capture_output:
                logger.debug(f"Command output: {result.stdout.strip()}")
                return result.stdout.strip()
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Command '{' '.join(command)}' failed with error: {e.stderr.strip() if e.stderr else str(e)}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command '{' '.join(command)}' timed out after {timeout} seconds")
            raise
    
    def validate_prerequisites(self) -> bool:
        """Validate that required tools and credentials are available."""
        logger.info("Validating prerequisites...")
        
        # Check required tools
        required_tools = ['terraform', 'ansible-playbook', 'az']
        for tool in required_tools:
            try:
                self.run_command(['which', tool], capture_output=True)
                logger.info(f"✓ {tool} is available")
            except subprocess.CalledProcessError:
                logger.error(f"✗ {tool} is not available in PATH")
                return False
        
        # Check Azure CLI login
        try:
            self.run_command(['az', 'account', 'show'], capture_output=True)
            logger.info("✓ Azure CLI is authenticated")
        except subprocess.CalledProcessError:
            logger.error("✗ Azure CLI is not authenticated. Run 'az login' first.")
            return False
        
        # Check required configuration
        required_configs = ['resource_group_name', 'resource_name_prefix']
        for config_key in required_configs:
            if not self.config.get(config_key):
                logger.error(f"✗ Required configuration '{config_key}' is missing")
                return False
        
        logger.info("✓ All prerequisites validated")
        return True
    
    def terraform_init(self):
        """Initialize Terraform."""
        logger.info("Initializing Terraform...")
        self.run_command(['terraform', 'init'], timeout=300)
        logger.info("✓ Terraform initialized")
    
    def terraform_plan(self) -> str:
        """Create and validate Terraform plan."""
        logger.info("Creating Terraform plan...")
        
        plan_vars = [
            '-var', f'resource_group_name={self.config["resource_group_name"]}',
            '-var', f'resource_name_prefix={self.config["resource_name_prefix"]}',
            '-var', f'resource_location={self.config["resource_location"]}',
            '-var', f'create_resource_group={str(self.config["create_resource_group"]).lower()}',
            '-var', f'username={self.config["username"]}'
        ]
        
        if self.config.get("subscription_id"):
            plan_vars.extend(['-var', f'subscription_id={self.config["subscription_id"]}'])
        
        plan_file = "main.tfplan"
        command = ['terraform', 'plan', '-out', plan_file] + plan_vars
        
        self.run_command(command, timeout=300)
        logger.info(f"✓ Terraform plan created: {plan_file}")
        return plan_file
    
    def terraform_apply(self, plan_file: str):
        """Apply Terraform plan to provision infrastructure."""
        logger.info("Applying Terraform plan...")
        self.run_command(['terraform', 'apply', plan_file], timeout=self.config["provisioning_timeout"])
        logger.info("✓ Infrastructure provisioned successfully")
    
    def get_terraform_outputs(self) -> Dict[str, str]:
        """Retrieve all Terraform outputs."""
        logger.info("Retrieving Terraform outputs...")
        
        output_json = self.run_command(['terraform', 'output', '-json'], capture_output=True)
        outputs = json.loads(output_json)
        
        # Extract values from Terraform output format
        extracted_outputs = {}
        for key, value in outputs.items():
            extracted_outputs[key] = value['value']
        
        logger.info("✓ Terraform outputs retrieved")
        return extracted_outputs
    
    def save_private_keys(self, outputs: Dict[str, str]) -> Dict[str, str]:
        """Save SSH private keys to files."""
        logger.info("Saving SSH private keys...")
        
        key_files = {}
        for i in [1, 2]:
            key_data = outputs[f'key_data_{i}']
            key_file = f"vm_{i}_private_key.pem"
            
            with open(key_file, 'w') as f:
                f.write(key_data)
            os.chmod(key_file, 0o600)
            
            key_files[f'vm_{i}'] = key_file
        
        logger.info("✓ SSH private keys saved")
        return key_files
    
    def generate_ansible_inventory(self, outputs: Dict[str, str]) -> str:
        """Generate Ansible inventory from Terraform outputs."""
        logger.info("Generating Ansible inventory...")
        
        inventory = {
            'all': {
                'vars': {
                    'k8s_user': outputs['username'],
                    'k8s_user2': outputs['username'],
                    'ansible_ssh_common_args': '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
                },
                'children': {
                    'control_nodes': {
                        'hosts': {
                            'control_node': {
                                'ansible_host': outputs['public_ip_address_1'],
                                'ansible_user': outputs['username'],
                                'ansible_ssh_private_key_file': f'../../terraform/vm_1_private_key.pem'
                            }
                        }
                    },
                    'worker_nodes': {
                        'hosts': {
                            'worker_node_1': {
                                'ansible_host': outputs['public_ip_address_2'],
                                'ansible_user': outputs['username'],
                                'ansible_ssh_private_key_file': f'../../terraform/vm_2_private_key.pem'
                            }
                        }
                    }
                }
            }
        }
        
        inventory_file = self.ansible_dir / self.config["ansible_inventory_file"]
        with open(inventory_file, 'w') as f:
            yaml.dump(inventory, f, default_flow_style=False)
        
        logger.info(f"✓ Ansible inventory generated: {inventory_file}")
        return str(inventory_file)
    
    def wait_for_ssh_connectivity(self, outputs: Dict[str, str], key_files: Dict[str, str]):
        """Wait for SSH connectivity to both VMs."""
        logger.info("Waiting for SSH connectivity...")
        
        hosts = [
            (outputs['public_ip_address_1'], key_files['vm_1'], 'controller'),
            (outputs['public_ip_address_2'], key_files['vm_2'], 'worker')
        ]
        
        for ip, key_file, name in hosts:
            logger.info(f"Waiting for {name} ({ip}) to be accessible...")
            max_attempts = 60
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    self.run_command([
                        'ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
                        '-o', 'ConnectTimeout=10', '-o', 'UserKnownHostsFile=/dev/null',
                        f"{outputs['username']}@{ip}", 'echo "SSH connection successful"'
                    ], capture_output=True, timeout=15)
                    logger.info(f"✓ {name} is accessible")
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    attempt += 1
                    if attempt < max_attempts:
                        time.sleep(10)
                    else:
                        raise Exception(f"Failed to establish SSH connection to {name} after {max_attempts} attempts")
        
        logger.info("✓ All VMs are accessible via SSH")
    
    def run_ansible_playbooks(self, inventory_file: str):
        """Execute Ansible playbooks to configure the cluster."""
        logger.info("Running Ansible playbooks...")
        
        playbooks = [
            ('setup_common.yml', 'Installing common prerequisites'),
            ('remote_setup_controller_worker.yml', 'Setting up Kubernetes cluster')
        ]
        
        for playbook, description in playbooks:
            logger.info(f"Running {description}...")
            playbook_path = self.ansible_dir / playbook
            
            if not playbook_path.exists():
                logger.error(f"Playbook not found: {playbook_path}")
                continue
            
            self.run_command([
                'ansible-playbook', 
                '-i', inventory_file,
                str(playbook_path)
            ], cwd=str(self.ansible_dir), timeout=1800)
            
            logger.info(f"✓ {description} completed")
        
        logger.info("✓ Ansible configuration completed")
    
    def setup_aiopslab(self, outputs: Dict[str, str], key_files: Dict[str, str]):
        """Setup AIOpsLab on the controller node."""
        logger.info("Setting up AIOpsLab...")
        
        controller_ip = outputs['public_ip_address_1']
        username = outputs['username']
        key_file = key_files['vm_1']
        
        # Copy and execute the setup script
        setup_script = self.terraform_dir / "scripts" / "setup_aiopslab.sh"
        
        # Copy script to remote
        self.run_command([
            'scp', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
            str(setup_script), f"{username}@{controller_ip}:~/setup_aiopslab.sh"
        ])
        
        # Execute script
        self.run_command([
            'ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
            f"{username}@{controller_ip}", 
            'bash ~/setup_aiopslab.sh'
        ], timeout=1800)
        
        logger.info("✓ AIOpsLab setup completed")
    
    def deploy(self):
        """Main deployment orchestration method."""
        try:
            logger.info("Starting AIOpsLab deployment...")
            
            # Validate prerequisites
            if not self.validate_prerequisites():
                logger.error("Prerequisites validation failed")
                return False
            
            # Terraform workflow
            self.terraform_init()
            plan_file = self.terraform_plan()
            self.terraform_apply(plan_file)
            
            # Get outputs and prepare for Ansible
            outputs = self.get_terraform_outputs()
            key_files = self.save_private_keys(outputs)
            
            # Wait for VMs to be ready
            self.wait_for_ssh_connectivity(outputs, key_files)
            
            # Generate inventory and run Ansible
            inventory_file = self.generate_ansible_inventory(outputs)
            self.run_ansible_playbooks(inventory_file)
            
            # Setup AIOpsLab
            self.setup_aiopslab(outputs, key_files)
            
            # Display connection information
            self._display_connection_info(outputs, key_files)
            
            logger.info("🎉 AIOpsLab deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            return False
    
    def _display_connection_info(self, outputs: Dict[str, str], key_files: Dict[str, str]):
        """Display connection information for the deployed infrastructure."""
        logger.info("\n" + "="*60)
        logger.info("DEPLOYMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"Controller Public IP: {outputs['public_ip_address_1']}")
        logger.info(f"Worker Public IP: {outputs['public_ip_address_2']}")
        logger.info("")
        logger.info("SSH Connection Commands:")
        logger.info(f"Controller: ssh -i {key_files['vm_1']} {outputs['username']}@{outputs['public_ip_address_1']}")
        logger.info(f"Worker:     ssh -i {key_files['vm_2']} {outputs['username']}@{outputs['public_ip_address_2']}")
        logger.info("")
        logger.info("To access AIOpsLab:")
        logger.info("1. SSH into the controller node")
        logger.info("2. cd ~/AIOpsLab && source .venv/bin/activate")
        logger.info("3. Follow the AIOpsLab documentation for usage")
        logger.info("="*60)
    
    def destroy(self):
        """Destroy the provisioned infrastructure."""
        logger.info("Destroying infrastructure...")
        try:
            self.run_command(['terraform', 'destroy', '-auto-approve'], timeout=600)
            logger.info("✓ Infrastructure destroyed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy infrastructure: {str(e)}")
            return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIOpsLab Unified Deployment Tool")
    parser.add_argument('action', choices=['deploy', 'destroy'], 
                       help='Action to perform')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--resource-group', required=True,
                       help='Azure resource group name')
    parser.add_argument('--prefix', required=True,
                       help='Resource name prefix')
    parser.add_argument('--location', default='westus2',
                       help='Azure region (default: westus2)')
    parser.add_argument('--create-resource-group', action='store_true',
                       help='Create new resource group')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create deployment configuration
    config = {
        'resource_group_name': args.resource_group,
        'resource_name_prefix': args.prefix,
        'resource_location': args.location,
        'create_resource_group': args.create_resource_group
    }
    
    deployment = AIOpsLabDeployment()
    deployment.config.update(config)
    
    if args.action == 'deploy':
        success = deployment.deploy()
        sys.exit(0 if success else 1)
    elif args.action == 'destroy':
        success = deployment.destroy()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()