## Environment Setup

This is the instruction to use Ansible to build a remote cluster for AIOpsLab. We currently use [CloudLab](https://www.cloudlab.us/) but we believe this will work on any servers you have access to.

There are two ways to setup AIOpsLab for a remote cluster, please choose either approach (A) or (B) to run AIOpsLab.

## (A) Run AIOpsLab inside the cluster
If you want to run your agent and AIOpsLab **inside** the cluster, please upload the _whole AIOpsLab codebase_ to the control node, and follow steps 1), 2), 3), 4).

### 1) Exchange SSH keys
**Do this on your own machine that is controlling the cluster.**
Your nodes need to be able to ssh into each other to setup the cluster. If your nodes have different keys then the device you're running AIOpsLab on, or if you're unsure, proceed with the following steps to exchange ssh keys between the nodes.

The nodes should exchange their public keys with each other before running Ansible.
In the machine that can access all the VMs (e.g., your laptop whose key has been stored in CloudLab), 
modify the `ssh/hosts.txt` file to record down IPs/hostnames of the nodes.
Then run the following commands:

```shell
cd ssh
./keys.sh
```

### 2) Install Ansible (Ubuntu)

SSH into your desired control node: `ssh <USER_NAME>@<HOST>`

Then run:
```shell
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common

sudo add-apt-repository --yes --update ppa:ansible/ansible

#Install Ansible
sudo apt install ansible -y

#Validate Ansible:
ansible --version 
```

Once those commands have completed, you can exit the SSH session.

## (B) Run AIOpsLab outside of the cluster
If you want to run your agent and AIOpsLab **outside** of the cluster (e.g., on your own workstation), please follow steps 3), 4) (1 and 2 are no longer needed).


### 3) Modify the inventory file
```bash
cp inventory.yml.example inventory.yml
```

Modify the IPs and user names in the inventory file accordingly, `inventory.yml`. 

### 4) Run the Ansible playbook

Install some common requirements (e.g., package installation) in all of the nodes; and setup the controllers and workers to run K8S:
```shell
ansible-playbook -i inventory.yml setup_common.yml && ansible-playbook -i inventory.yml remote_setup_controller_worker.yml
```

After these, you should see every node running inside a K8S cluster:
```shell
kubectl get nodes
```

### Common Errors
If you're running into issues from Ansible related to host key authentication, try typing `yes` in your terminal for each node, or proceeding with the following steps:

You can create a file in the same directory as this README called `ansible.cfg` to turn off that warning:
```yaml
[defaults]
host_key_checking = False
```
Be mindful about the security implications of disabling host key checking, if you're not aware ask someone who is.
