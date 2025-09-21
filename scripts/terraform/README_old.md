
## Setting up AIOpsLab using Terraform

This guide outlines the steps for establishing a secure connection to your Azure environment using a VPN and then provisioning resources with Terraform. This will create a two-node Kubernetes cluster with one controller and one worker node.

**NOTE**: This will incur cloud costs as resources are created on Azure.

**Prerequisites:**

- **Azure VPN Connection:** Set up a secure connection to your Azure environment using a VPN client.
- **Working directory:** AIOpsLab/scripts/terraform/
- **Privileges:** The user should have the privileges to create resources (SSH keys, VM, network interface, network interface security group (if required), public IP, subnet, virtual network) in the selected resource group.
- **Azure CLI:** Follow the official [Microsoft documentation](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) for installing the Azure CLI for your operating system: 
- **Install and initialize Terraform:**
  
     a. Download and install Terraform from the [official HashiCorp website](https://developer.hashicorp.com/terraform/install);
  
     b. To make the initial dependency selections that will initialize the dependency lock file, run:
   
      terraform init
  
**Steps:**
   
1. **Authenticate with Azure CLI**

   Open a terminal window and run the following command to log in to Azure:

   ```shell
   az login
   ```

2. **Select subscription**

   The output of az login will have a list of subscriptions you have access to. Copy the value in the "id" column of the subscription you want to work with:
   
   ```shell
   az account set --subscription "<id>"
   ```
3. **Verify the plan**

   *Note*: The SSH port of the VMs is open to the public. Please update the NSG resource in the main.tf file to restrict incoming traffic. Use the source_address_prefix attribute to specify allowed sources (e.g., source_address_prefix = "CorpNetPublic").

   Create and save the plan by passing the required variables

   a) _resource_group_name_ (rg): the resource group where the resources would be created.

   b) _resource_prefix_name_ (prefix): a prefix for all the resources created using the Terraform script.

   ```shell
   terraform plan -out main.tfplan -var " resource_group_name=<rg>" -var "resource_name_prefix=<prefix>"
   ```
5. **Apply the saved plan**

   Note: Verify the plan from the previous step before applying it.

   ```shell
   terraform apply "main.tfplan"
   ```
   
6. **Setup AIOpsLab**
    Run the below script to setup AIOpsLab on the newly provisioned resources

    ```shell
    python deploy.py
    ```
   On successful execution, the script outputs the SSH commands to login to the controller and worker node. Please save it.

   Please activate virtual environment before running any scripts and add the path to `wrk2` executable to PATH:

   ```
   azureuser@kubeController:~/AIOpsLab$ source .venv/bin/activate
   (.venv) azureuser@kubeController:~/AIOpsLab/clients$ export PATH="$PATH:/home/azureuser/AIOpsLab/TargetMicroservices/wrk2"
   ```

**How to destroy the resources using Terraform?**

1. Before deleting the resources, run the below command to create and save a plan (use the values previous used for resource_group_name and resource_name_prefix)
   
    ```shell
    terraform plan -destroy -out main.destroy.tfplan -var "resource_group_name=<rg>" -var "resource_name_prefix=<prefix>"
    ```

2. Once the plan is verified, remove the resources using the below command:

    ```shell
    terraform destroy main.destroy.tfplan
    ```

