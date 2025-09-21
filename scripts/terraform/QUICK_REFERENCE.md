# AIOpsLab Automation Quick Reference

## 🚀 Quick Start
```bash
cd AIOpsLab/scripts/terraform/
az login
terraform init
python deploy.py
```

## 🔧 Advanced Usage

### Deploy with specific parameters
```bash
python deploy_unified.py deploy \
  --resource-group "my-aiopslab-rg" \
  --prefix "aiopslab" \
  --location "westus2" \
  --create-resource-group
```

### Deploy with configuration file
```bash
cp config.yml.example config.yml
# Edit config.yml with your settings
python deploy_unified.py deploy --config config.yml
```

### Verify deployment
```bash
python verify_deployment.py
```

### Destroy infrastructure
```bash
python deploy_unified.py destroy \
  --resource-group "my-aiopslab-rg" \
  --prefix "aiopslab"
```

## 📁 File Structure
```
scripts/terraform/
├── deploy.py              # Simple deployment wrapper
├── deploy_unified.py      # Advanced deployment orchestration
├── verify_deployment.py   # Post-deployment verification
├── config.yml.example    # Configuration template
├── test_deployment.py     # Automated tests
├── main.tf               # Core infrastructure definition
├── variables.tf          # Input variables
├── outputs.tf            # Output values
├── data.tf              # Data sources and locals
├── ssh.tf               # SSH key management
└── providers.tf         # Provider configuration
```

## ⚡ What's Automated
- ✅ Azure infrastructure provisioning
- ✅ SSH key generation and management
- ✅ Ansible inventory generation
- ✅ Kubernetes cluster setup
- ✅ AIOpsLab installation
- ✅ Network connectivity validation
- ✅ Error handling and retries
- ✅ Post-deployment verification

## 🔍 Troubleshooting

### Common Issues
- **SSH connectivity**: Wait 2-3 minutes after deployment
- **Ansible failures**: Check VM accessibility with `verify_deployment.py`
- **Terraform state**: Use `terraform refresh` if needed
- **Kubernetes issues**: SSH to nodes and check `systemctl status kubelet`

### Debug Commands
```bash
# Check deployment status
python verify_deployment.py

# View Terraform outputs
terraform output

# Test SSH connectivity
ssh -i vm_1_private_key.pem azureuser@<controller-ip>

# Check Kubernetes cluster
kubectl get nodes
```

## 🔐 Security Notes
- SSH keys are auto-generated and stored locally
- VM SSH ports are open to public by default
- Update NSG rules in `main.tf` for production use
- Delete private key files after use if not needed

## 📞 Support
For issues and questions:
1. Check the troubleshooting section in README.md
2. Run `verify_deployment.py` for detailed diagnostics
3. Review logs from the deployment script
4. Check Azure portal for resource status