#!/usr/bin/env python3
"""
Post-deployment verification script for AIOpsLab.
This script connects to the deployed infrastructure and validates that everything is working correctly.
"""

import subprocess
import sys
import json
import time
from pathlib import Path


def run_command(command, capture_output=False, timeout=30):
    """Execute a shell command with error handling."""
    try:
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=True,
            timeout=timeout
        )
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(command)}")
        print(f"   Error: {e.stderr.strip() if e.stderr else str(e)}")
        return None
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out: {' '.join(command)}")
        return None


def get_terraform_outputs():
    """Get Terraform outputs."""
    print("📋 Retrieving deployment information...")
    
    try:
        output_json = run_command(['terraform', 'output', '-json'], capture_output=True)
        if not output_json:
            return None
            
        outputs = json.loads(output_json)
        extracted_outputs = {}
        for key, value in outputs.items():
            extracted_outputs[key] = value['value']
            
        return extracted_outputs
    except Exception as e:
        print(f"❌ Failed to get Terraform outputs: {e}")
        return None


def verify_ssh_connectivity(host, key_file, username):
    """Verify SSH connectivity to a host."""
    print(f"🔑 Testing SSH connectivity to {host}...")
    
    result = run_command([
        'ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
        '-o', 'ConnectTimeout=10', '-o', 'UserKnownHostsFile=/dev/null',
        f'{username}@{host}', 'echo "SSH connection successful"'
    ], capture_output=True, timeout=15)
    
    return result is not None


def verify_kubernetes_cluster(controller_host, key_file, username):
    """Verify that Kubernetes cluster is running."""
    print("☸️  Verifying Kubernetes cluster...")
    
    # Check if kubectl works
    result = run_command([
        'ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
        f'{username}@{controller_host}',
        'kubectl get nodes'
    ], capture_output=True, timeout=30)
    
    if result:
        print("✅ Kubernetes cluster is running")
        print("   Node status:")
        for line in result.split('\n'):
            print(f"   {line}")
        return True
    else:
        print("❌ Kubernetes cluster is not responding")
        return False


def verify_aiopslab_installation(controller_host, key_file, username):
    """Verify that AIOpsLab is installed."""
    print("🔬 Verifying AIOpsLab installation...")
    
    # Check if AIOpsLab directory exists and virtual environment is set up
    result = run_command([
        'ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
        f'{username}@{controller_host}',
        'test -d ~/AIOpsLab && test -f ~/AIOpsLab/.venv/bin/activate && echo "AIOpsLab installed"'
    ], capture_output=True, timeout=15)
    
    if result:
        print("✅ AIOpsLab is installed")
        return True
    else:
        print("❌ AIOpsLab installation not found")
        return False


def verify_services(controller_host, key_file, username):
    """Verify that essential services are running."""
    print("🔧 Verifying essential services...")
    
    services = ['docker', 'kubelet', 'cri-docker']
    all_services_ok = True
    
    for service in services:
        result = run_command([
            'ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
            f'{username}@{controller_host}',
            f'sudo systemctl is-active {service}'
        ], capture_output=True, timeout=10)
        
        if result and result.strip() == 'active':
            print(f"   ✅ {service} is running")
        else:
            print(f"   ❌ {service} is not running")
            all_services_ok = False
    
    return all_services_ok


def verify_network_connectivity(controller_host, worker_host, key_file, username):
    """Verify network connectivity between nodes."""
    print("🌐 Verifying network connectivity between nodes...")
    
    # Test ping from controller to worker
    result = run_command([
        'ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
        f'{username}@{controller_host}',
        f'ping -c 3 {worker_host}'
    ], capture_output=True, timeout=20)
    
    if result and 'bytes from' in result:
        print("✅ Network connectivity between nodes is working")
        return True
    else:
        print("❌ Network connectivity issues detected")
        return False


def main():
    """Main verification function."""
    print("🚀 AIOpsLab Deployment Verification")
    print("=" * 50)
    
    # Change to terraform directory
    terraform_dir = Path(__file__).parent
    import os
    os.chdir(terraform_dir)
    
    # Get deployment outputs
    outputs = get_terraform_outputs()
    if not outputs:
        print("❌ Could not retrieve deployment information")
        sys.exit(1)
    
    controller_ip = outputs.get('public_ip_address_1')
    worker_ip = outputs.get('public_ip_address_2')
    username = outputs.get('username', 'azureuser')
    
    print(f"📊 Deployment Details:")
    print(f"   Controller IP: {controller_ip}")
    print(f"   Worker IP: {worker_ip}")
    print(f"   Username: {username}")
    print()
    
    # Check if key files exist
    controller_key = 'vm_1_private_key.pem'
    worker_key = 'vm_2_private_key.pem'
    
    if not Path(controller_key).exists():
        print(f"❌ SSH key file not found: {controller_key}")
        sys.exit(1)
    
    if not Path(worker_key).exists():
        print(f"❌ SSH key file not found: {worker_key}")
        sys.exit(1)
    
    # Run verification tests
    verification_results = []
    
    # Test SSH connectivity
    verification_results.append((
        "SSH to Controller",
        verify_ssh_connectivity(controller_ip, controller_key, username)
    ))
    
    verification_results.append((
        "SSH to Worker",
        verify_ssh_connectivity(worker_ip, worker_key, username)
    ))
    
    # Test services
    verification_results.append((
        "Essential Services",
        verify_services(controller_ip, controller_key, username)
    ))
    
    # Test Kubernetes
    verification_results.append((
        "Kubernetes Cluster",
        verify_kubernetes_cluster(controller_ip, controller_key, username)
    ))
    
    # Test AIOpsLab installation
    verification_results.append((
        "AIOpsLab Installation",
        verify_aiopslab_installation(controller_ip, controller_key, username)
    ))
    
    # Test network connectivity
    private_worker_ip = outputs.get('private_ip_address_2', worker_ip)
    verification_results.append((
        "Network Connectivity",
        verify_network_connectivity(controller_ip, private_worker_ip, controller_key, username)
    ))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Verification Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in verification_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All verification tests passed!")
        print("Your AIOpsLab deployment is ready to use.")
        print()
        print("To get started:")
        print(f"1. SSH into the controller: ssh -i {controller_key} {username}@{controller_ip}")
        print("2. Activate the environment: cd ~/AIOpsLab && source .venv/bin/activate")
        print("3. Follow the AIOpsLab documentation for usage instructions")
    else:
        print("⚠️  Some verification tests failed.")
        print("Please check the output above and troubleshoot the issues.")
        print("You may need to wait a few more minutes for services to fully start.")
        sys.exit(1)


if __name__ == '__main__':
    main()