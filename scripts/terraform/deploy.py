#!/usr/bin/env python3
"""
Legacy deploy.py - now serves as a simple wrapper around the unified deployment tool.
For advanced usage, use deploy_unified.py directly.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """
    Simple wrapper that maintains compatibility with the original deploy.py.
    This will prompt for required parameters and use the unified deployment tool.
    """
    print("AIOpsLab Deployment Tool")
    print("========================")
    print()
    
    # Check if we can use the unified deployment tool
    unified_script = Path(__file__).parent / "deploy_unified.py"
    if not unified_script.exists():
        print("Error: Unified deployment script not found!")
        sys.exit(1)
    
    # Get required parameters
    print("This tool will provision Azure infrastructure and configure AIOpsLab.")
    print("Please provide the following information:")
    print()
    
    resource_group = input("Resource Group Name: ").strip()
    if not resource_group:
        print("Error: Resource group name is required!")
        sys.exit(1)
    
    prefix = input("Resource Name Prefix: ").strip()
    if not prefix:
        print("Error: Resource name prefix is required!")
        sys.exit(1)
    
    location = input("Azure Region [westus2]: ").strip() or "westus2"
    
    create_rg = input("Create new resource group? [y/N]: ").strip().lower()
    create_resource_group = create_rg in ['y', 'yes']
    
    print()
    print("Configuration Summary:")
    print(f"  Resource Group: {resource_group}")
    print(f"  Prefix: {prefix}")
    print(f"  Location: {location}")
    print(f"  Create RG: {create_resource_group}")
    print()
    
    confirm = input("Proceed with deployment? [y/N]: ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Deployment cancelled.")
        sys.exit(0)
    
    # Build command for unified deployment tool
    cmd = [
        sys.executable,
        str(unified_script),
        'deploy',
        '--resource-group', resource_group,
        '--prefix', prefix,
        '--location', location
    ]
    
    if create_resource_group:
        cmd.append('--create-resource-group')
    
    print()
    print("Starting deployment...")
    print()
    
    # Execute the unified deployment tool
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("Deployment completed successfully!")
    except subprocess.CalledProcessError as e:
        print()
        print(f"Deployment failed with exit code {e.returncode}")
        print("Check the logs above for details.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print()
        print("Deployment interrupted by user.")
        sys.exit(1)


if __name__ == '__main__':
    main()


if __name__ == "__main__":
    main()
