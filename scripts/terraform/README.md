# AIOpsLab Automated Deployment with Terraform + Ansible

**Fully automated deployment of production-ready Kubernetes clusters on Azure**

## 🚀 Quick Start

Deploy AIOpsLab with 3 workers in just one command:

```bash
python deploy.py --apply --workers 3 --vm-size Standard_D4s_v3 \
    --resource-group aiopslab-rg \
    --ssh-key ~/.ssh/id_rsa.pub
```

Destroy everything when done:

```bash
python deploy.py --destroy --resource-group aiopslab-rg --ssh-key ~/.ssh/id_rsa.pub
```

**That's it!** The script handles everything: VM provisioning, Kubernetes setup, and configuration.

---

## ✨ What's New (v2.0)

- ✅ **Fully Automated**: One command deploys everything
- ✅ **Dynamic Scaling**: Support for 1-N worker nodes
- ✅ **Ansible Integration**: Production-ready K8s setup
- ✅ **Smart Inventory**: Auto-generates Ansible inventory from Terraform
- ✅ **SSH Verification**: Waits for connectivity before proceeding
- ✅ **Graceful Destroy**: Safe teardown with confirmation
- ✅ **Better Outputs**: Structured VM information for automation

---

## 📋 Prerequisites

### 1. Software Requirements

| Tool | Version | Installation |
|------|---------|--------------|
| Python | 3.11+ | [python.org](https://python.org) |
| Terraform | 1.6+ | [Install](https://developer.hashicorp.com/terraform/install) |
| Ansible | Latest | [Install](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) |
| Azure CLI | Latest | [Install](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) |

#### Quick Install (Ubuntu/Debian)
```bash
# Ansible
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository --yes --update ppa:ansible/ansible
sudo apt install ansible -y

# Python dependencies
pip install pyyaml
```

### 2. Azure Setup

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription "<subscription-id>"

# Create resource group (if needed)
az group create --name aiopslab-rg --location eastus

# Generate SSH key (if needed)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
```

### 3. Initialize Terraform

```bash
cd scripts/terraform
terraform init
```

---

## 🎯 Usage

### Option 1: Automated Deployment with deploy.py

Deploy with default settings (2 workers, Standard_B2s):

```bash
python deploy.py --apply
```

### Custom Deployment

Specify worker count and VM size:

```bash
python deploy.py --apply \
    --workers 5 \
    --vm-size Standard_D8s_v3 \
    --resource-group my-rg \
    --prefix myaiops \
    --ssh-key ~/.ssh/custom_key.pub
```

### Available Options

```
--apply                Deploy infrastructure and setup cluster
--destroy              Destroy all infrastructure
--workers N            Number of worker nodes (default: 2)
--vm-size SIZE         Azure VM size (default: Standard_B2s)
--resource-group RG    Azure resource group (default: aiopslab-rg)
--prefix PREFIX        Resource name prefix (default: aiopslab)
--ssh-key PATH         SSH public key path (default: ~/.ssh/id_rsa.pub)
--debug                Enable debug logging
```

### Destroy Infrastructure

```bash
python deploy.py --destroy \
    --resource-group aiopslab-rg \
    --ssh-key ~/.ssh/id_rsa.pub
```

You'll be prompted to confirm before deletion.

---

### Option 2: Manual Step-by-Step Deployment

For more control or debugging, you can run each step manually:

#### Step 1: Provision Azure VMs with Terraform

```bash
cd scripts/terraform
terraform init
terraform plan -var="resource_group_name=<your-rg>" -var="cluster_size=3"
terraform apply -var="resource_group_name=<your-rg>" -var="cluster_size=3"
```

#### Step 2: Generate Ansible Inventory

```bash
python generate_inventory.py
# This creates ../ansible/inventory.yml with VM IPs and SSH config
```

#### Step 3: Run Ansible Playbooks

```bash
cd ../ansible

# Install Docker, Kubernetes packages on all nodes
ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i inventory.yml setup_common.yml

# Initialize K8s cluster and join workers
ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i inventory.yml remote_setup_controller_worker.yml
```

#### Step 4: Verify Cluster

```bash
# The playbook copies kubeconfig to your ~/.kube/config automatically
kubectl get nodes
```

#### Destroy Manually

```bash
cd scripts/terraform
terraform destroy -var="resource_group_name=<your-rg>"
```

---

## 🖥️ Mode A vs Mode B Deployment

### Mode A: AIOpsLab Inside Cluster (Recommended for full functionality)

Run AIOpsLab directly on the controller VM:

```bash
# SSH to controller
ssh -i ~/.ssh/id_rsa azureuser@<controller-public-ip>

# Clone and setup AIOpsLab
git clone --recurse-submodules https://github.com/microsoft/AIOpsLab.git
cd AIOpsLab
poetry install && poetry shell

# Configure for localhost
cd aiopslab
cp config.yml.example config.yml
# Set k8s_host=localhost in config.yml
```

**Pros**: All fault injectors work, no Docker required locally
**Cons**: Must SSH to controller to run experiments

### Mode B: AIOpsLab on Your Laptop (Convenient for development)

Run AIOpsLab locally, connecting to remote K8s cluster:

```bash
# Kubeconfig is automatically copied to ~/.kube/config by Ansible
kubectl get nodes  # Should show your remote cluster

# Configure AIOpsLab
cd aiopslab
cp config.yml.example config.yml
# Set k8s_host=<controller-hostname> and k8s_user=azureuser
```

**Pros**: Use local IDE, no SSH needed for running experiments
**Cons**: Some fault injectors (e.g., VirtualizationFaultInjector) require local Docker

**Note**: If you see Docker connection errors in Mode B, either install Docker on your laptop or switch to Mode A.

---

## 📊 VM Sizing Guide

| VM Size | vCPUs | RAM | Use Case | Cost/Month* |
|---------|-------|-----|----------|-------------|
| Standard_B2s | 2 | 4 GB | Dev/Test | ~$30 |
| Standard_D4s_v3 | 4 | 16 GB | Small Prod | ~$120 |
| Standard_D8s_v3 | 8 | 32 GB | Medium Prod | ~$240 |
| Standard_D16s_v3 | 16 | 64 GB | Large Prod | ~$480 |

*Approximate costs for East US region

---

## 🔧 What Gets Deployed

### Infrastructure (Terraform)
- 1 Controller VM (Kubernetes control plane)
- N Worker VMs (configurable, default 2)
- Virtual Network & Subnet (10.0.0.0/16)
- Network Security Group (SSH access)
- Public IPs for all VMs
- Network Interfaces

### Software Stack (Ansible)
- Docker CE + cri-dockerd
- Kubernetes v1.31 (kubeadm, kubelet, kubectl)
- Flannel CNI plugin
- Fully configured K8s cluster

---

## 📝 Deployment Workflow

```
1. Terraform Init      → Initialize providers
2. Terraform Plan      → Create execution plan
3. Terraform Apply     → Provision VMs on Azure
4. Get Outputs         → Retrieve VM IPs and config
5. Generate Inventory  → Create Ansible inventory.yml
6. Wait for SSH        → Ensure VMs are accessible
7. Run Ansible         → Install Docker, K8s packages
8. Setup Cluster       → Initialize K8s, join workers
9. Display Info        → Show SSH commands
```

**Total Time:** 15-25 minutes

---

## ✅ Post-Deployment

After deployment completes, you'll see:

```
================================================================================
🎉 DEPLOYMENT COMPLETE!
================================================================================

📋 Controller Node:
   Public IP:  20.123.45.67
   SSH:        ssh -i ~/.ssh/id_rsa azureuser@20.123.45.67

👷 Worker Nodes (3):
   Worker 1: 20.123.45.68
   Worker 2: 20.123.45.69
   Worker 3: 20.123.45.70
```

### Verify Cluster

```bash
# SSH into controller
ssh -i ~/.ssh/id_rsa azureuser@<controller-ip>

# Check nodes
kubectl get nodes

# Expected output:
# NAME                STATUS   ROLES           AGE   VERSION
# aiopslab-controller Ready    control-plane   5m    v1.31.x
# aiopslab-worker-1   Ready    <none>          3m    v1.31.x
# aiopslab-worker-2   Ready    <none>          3m    v1.31.x
```

### Deploy AIOpsLab

```bash
# On controller VM
cd ~
git clone --recurse-submodules https://github.com/microsoft/AIOpsLab.git
cd AIOpsLab

# Setup
poetry install
poetry shell

# Configure
cd aiopslab
cp config.yml.example config.yml
# Edit config.yml: set k8s_host=localhost

# Test
python3 cli.py
```

---

## 🐛 Troubleshooting

### SSH Connection Timeout

**Symptoms**: Deployment hangs at "Waiting for SSH"

**Solutions**:
1. Check Network Security Group allows your IP
2. Verify SSH key path is correct
3. Wait longer (VMs may be slow to boot)

```bash
# Test SSH manually
ssh -i ~/.ssh/id_rsa -v azureuser@<vm-ip>
```

### Ansible Playbook Fails

**Solutions**: Re-run Ansible manually:

```bash
cd scripts/ansible

# Run common setup
ansible-playbook -i inventory.yml setup_common.yml

# Run cluster setup
ansible-playbook -i inventory.yml remote_setup_controller_worker.yml
```

### Nodes Not Ready

**Solution**: Check Flannel CNI:

```bash
kubectl get pods -n kube-system | grep flannel

# If not running, reapply:
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
```

### kubeadm init fails with "conntrack not found"

**Cause**: Missing conntrack package (required for kube-proxy)

**Solution**: The setup_common.yml playbook should install this. If running manually:

```bash
sudo apt install conntrack -y
```

### kubectl from laptop shows certificate error

**Symptom**: `Unable to connect to the server: x509: certificate is valid for X, not Y`

**Cause**: K8s API server certificate doesn't include the public IP

**Solution**: The Ansible playbook automatically adds `--apiserver-cert-extra-sans` with the public IP. If you need to reinitialize:

```bash
# On controller, reset and reinit with SANs
sudo kubeadm reset -f
sudo kubeadm init \
  --pod-network-cidr=10.244.0.0/16 \
  --cri-socket unix:///var/run/cri-dockerd.sock \
  --apiserver-advertise-address=<private-ip> \
  --apiserver-cert-extra-sans=<public-ip>,<private-ip>
```

### Helm chart not found error

**Symptom**: `FileNotFoundError: Helm chart not found at: ...`

**Solution**: Clone with submodules:

```bash
git submodule update --init --recursive
```

### Docker connection error in Mode B

**Symptom**: `Error while fetching server API version: HTTPConnection.request() got an unexpected keyword argument 'chunked'`

**Cause**: Some fault injectors try to connect to local Docker daemon

**Solution**:
1. Install Docker Desktop on your laptop, OR
2. Use Mode A (run AIOpsLab on controller VM)

---

## 📚 Documentation

- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** - Comprehensive deployment guide
- **[SECURITY.md](./SECURITY.md)** - Security best practices and considerations
- **[DEPLOYMENT_VALIDATION.md](../../DEPLOYMENT_VALIDATION.md)** - Architecture and validation report
- **[../ansible/README.md](../ansible/README.md)** - Ansible playbook documentation

---

## 🔐 Security Notes

**⚠️ READ BEFORE PRODUCTION DEPLOYMENT: [SECURITY.md](./SECURITY.md)**

### Default Behavior (Secure)

The deployment script is **secure by default**:
- ✅ Automatically adds SSH host keys to prevent MITM attacks
- ✅ Host key verification is ENABLED
- ✅ Only use `--disable-host-key-checking` for testing/CI

### Quick Security Checklist

- [ ] **NSG Rules:** SSH is open to 0.0.0.0/0 by default - restrict it!
  ```bash
  # Use --restrict-ssh-corpnet for Microsoft CorpNet
  python deploy.py --apply --workers 2 --restrict-ssh-corpnet

  # Or add custom IP after deployment
  az network nsg rule create -g aiopslab-rg --nsg-name aiopslab-nsg \
      --name SSH-MyIP --priority 100 --protocol TCP \
      --source-address-prefixes "YOUR_IP/32" --destination-port-ranges 22
  ```

- [ ] **SSH Keys:** Use 4096-bit RSA or Ed25519 with passphrases
- [ ] **Production:** Consider Azure Bastion for secure access
- [ ] **Environments:** Use separate resource groups for prod/dev/test

**See [SECURITY.md](./SECURITY.md) for complete security documentation.**

---

## 💰 Cost Management

### Estimated Costs

**Small Dev Setup** (2 workers, B2s): ~$90/month
**Medium Prod** (3 workers, D4s_v3): ~$480/month
**Large Prod** (5 workers, D8s_v3): ~$1,440/month

### Save Money

1. Destroy when not in use: `python deploy.py --destroy`
2. Use B-series VMs for dev/test
3. Deallocate VMs instead of deleting:
   ```bash
   az vm deallocate --resource-group aiopslab-rg --name aiopslab-controller
   ```

---

## 📦 Files

```
scripts/terraform/
├── deploy.py                    # Main deployment script (NEW)
├── deploy_old.py                # Original script (backup)
├── generate_inventory.py        # Inventory generator (NEW)
├── main.tf                      # Infrastructure definition
├── variables.tf                 # Configuration variables
├── outputs.tf                   # Outputs (UPDATED)
├── providers.tf                 # Provider configuration
├── terraform.tfvars.example     # Config template (NEW)
├── DEPLOYMENT_GUIDE.md          # Full guide (NEW)
└── README.md                    # This file
```

---

## 🔄 Migration from v1.0

If you were using the old deployment method:

1. **Backup your old deploy.py**: Already saved as `deploy_old.py`
2. **Update Terraform**: Run `terraform init -upgrade`
3. **Use new deploy.py**: Follow Quick Start above
4. **Note**: Old deployments are incompatible with new outputs

---

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue or submit a PR!

Areas for improvement:
- Support for AWS, GCP
- Automated monitoring setup
- Cost optimization features
- Integration tests

---

## 📄 License

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

---

**Need Help?** See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) or open an issue.
