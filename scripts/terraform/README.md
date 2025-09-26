## Setting up AIOpsLab using Terraform and Ansible

This guide outlines the automated steps for provisioning Azure infrastructure using Terraform and configuring AIOpsLab using Ansible. The process has been significantly streamlined to reduce manual steps.

**NOTE**: This will incur cloud costs as resources are created on Azure.

### Prerequisites

- **Azure CLI:** Follow the official [Microsoft documentation](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) for installing the Azure CLI
- **Terraform:** Download and install from the [official HashiCorp website](https://developer.hashicorp.com/terraform/install)
- **Ansible:** Install using your package manager (e.g., `sudo apt install ansible` on Ubuntu)
- **Python 3.11+** with PyYAML: `pip install PyYAML`
- **Azure VPN Connection:** Set up a secure connection to your Azure environment using a VPN client
- **Privileges:** The user should have privileges to create resources (SSH keys, VMs, networking, storage) in Azure

### Quick Start (Recommended)

1. **Authenticate with Azure CLI**
   ```shell
   az login
   az account set --subscription "<your-subscription-id>"
   ```

2. **Navigate to the Terraform directory**
   ```shell
   cd AIOpsLab/scripts/terraform/
   ```

3. **Initialize Terraform**
   ```shell
   terraform init
   ```

4. **Run the simplified deployment script**
   ```shell
   python deploy.py
   ```
   
   The script will prompt you for:
   - Resource group name
   - Resource name prefix  
   - Azure region (default: westus2)
   - Whether to create a new resource group

5. **Wait for completion**
   The script will automatically:
   - Provision Azure infrastructure using Terraform
   - Generate Ansible inventory from Terraform outputs
   - Install and configure Kubernetes using Ansible
   - Set up AIOpsLab on the controller node
   - Display SSH connection information

### Advanced Usage

For more control over the deployment process, use the unified deployment tool directly:

```shell
# Deploy with specific parameters
python deploy_unified.py deploy \
  --resource-group "my-aiopslab-rg" \
  --prefix "aiopslab" \
  --location "westus2" \
  --create-resource-group

# Destroy infrastructure
python deploy_unified.py destroy \
  --resource-group "my-aiopslab-rg" \
  --prefix "aiopslab"

# Use configuration file
cp config.yml.example config.yml
# Edit config.yml with your settings
python deploy_unified.py deploy --config config.yml
```

### Manual Steps (Legacy Process)

If you prefer the manual approach or need to troubleshoot, you can still use the individual components:

1. **Create Terraform plan**
   ```shell
   terraform plan -out main.tfplan \
     -var "resource_group_name=<rg>" \
     -var "resource_name_prefix=<prefix>"
   ```

2. **Apply the plan**
   ```shell
   terraform apply "main.tfplan"
   ```

3. **Generate Ansible inventory**
   ```shell
   # The unified script does this automatically, but you can generate manually:
   terraform output -json > outputs.json
   # Then create inventory.yml based on the outputs
   ```

4. **Run Ansible playbooks**
   ```shell
   cd ../ansible/
   ansible-playbook -i inventory.yml setup_common.yml
   ansible-playbook -i inventory.yml remote_setup_controller_worker.yml
   ```

### Post-Deployment

After successful deployment:

1. **SSH into the controller node**
   ```shell
   ssh -i vm_1_private_key.pem azureuser@<controller-public-ip>
   ```

2. **Activate the AIOpsLab environment**
   ```shell
   cd ~/AIOpsLab
   source .venv/bin/activate
   ```

3. **Verify Kubernetes cluster**
   ```shell
   kubectl get nodes
   ```

### Cleanup

To destroy the infrastructure:

```shell
# Using the unified tool
python deploy_unified.py destroy --resource-group "my-aiopslab-rg" --prefix "aiopslab"

# Or manually
terraform destroy -auto-approve
```

### Security Notes

- The SSH port of the VMs is open to the public by default. Update the NSG resources in main.tf to restrict incoming traffic using the `source_address_prefix` attribute (e.g., `source_address_prefix = "CorpNetPublic"`)
- SSH private keys are generated automatically and stored locally. Keep them secure and delete them after use if not needed
- Consider using Azure Key Vault for production deployments

### Troubleshooting

- **SSH connectivity issues**: Wait a few minutes after deployment for VMs to fully initialize
- **Ansible playbook failures**: Check that all VMs are accessible and have the correct SSH keys
- **Terraform state issues**: Use `terraform refresh` to update state if resources were modified outside Terraform
- **Kubernetes issues**: SSH into nodes and check service status with `systemctl status kubelet`

### What's Automated

The enhanced deployment process now automates:
- ✅ Azure resource provisioning (VMs, networking, storage)
- ✅ SSH key generation and management
- ✅ Ansible inventory generation from Terraform outputs
- ✅ Kubernetes cluster setup and configuration
- ✅ AIOpsLab installation and configuration
- ✅ Network connectivity validation
- ✅ Error handling and progress reporting
- ✅ Infrastructure cleanup/destruction

This reduces the manual steps from ~15-20 individual commands to a single deployment command!