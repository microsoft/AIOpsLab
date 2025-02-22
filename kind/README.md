# AIOpsLab Deployment on WSL2 (Ubuntu)

This document provides detailed, step-by-step instructions for deploying AIOpsLab in an Ubuntu environment running on WSL2. It covers all prerequisites (installing Docker, kind, kubectl, Lua, Luarocks, and Luasocket), building the custom KIND image, creating a local Kubernetes cluster with kind, and deploying the AIOpsLab application.

---

## Table of Contents

- [Overview](#overview)
- [System Compatibility](#system-compatibility)
- [Prerequisites](#prerequisites)
  - [WSL2 and Ubuntu Setup](#wsl2-and-ubuntu-setup)
  - [Install Docker](#install-docker)
  - [Install kind](#install-kind)
  - [Install kubectl](#install-kubectl)
  - [Install Helm](#install-helm)
  - [Install Lua, Luarocks, and Luasocket](#install-lua-luarocks-and-luasocket)
- [Deployment Steps](#deployment-steps)
  - [1. Build the Custom KIND Image](#1-build-the-custom-kind-image)
  - [2. (Optional) Push the Image to Dockerhub](#2-optional-push-the-image-to-dockerhub)
  - [3. Create persistent directories for Prometheus data](#3-create-persistent-directories-for-prometheus-data)
  - [4. Create a KIND Kubernetes Cluster](#4-create-a-kind-kubernetes-cluster)
- [Troubleshooting](#troubleshooting)
- [Conclusion](#conclusion)

---

## **Overview**
AIOpsLab is deployed using **containerized components** and Kubernetes manifests. This guide provides a step-by-step deployment process, covering:

- Setting up **WSL2 (Windows Subsystem for Linux) or native Ubuntu 24.04**.
- Installing **Docker, kind, kubectl, Helm, Lua, Luarocks, and Luasocket**.
- Building a custom **KIND image** and deploying AIOpsLab into a **local Kubernetes cluster**.

---

## **System Compatibility**
This setup has been successfully verified on the following environments:
1. **WSL2 Ubuntu**  
   ```
   Linux Warrens-Laptop 5.15.167.4-microsoft-standard-WSL2 #1 SMP Tue Nov 5 00:21:55 UTC 2024 x86_64 GNU/Linux
   ```
2. **Ubuntu 24.04 LTS (Cloud/Local Machine)**  
   ```
   Linux ubuntu-s-4vcpu-8gb-sfo3-01 6.8.0-52-generic #53-Ubuntu SMP PREEMPT_DYNAMIC Sat Jan 11 00:06:25 UTC 2025 x86_64 GNU/Linux
   ```

---

## **Prerequisites**

### **WSL2 and Ubuntu Setup**
Ensure that WSL2 is enabled on Windows and Ubuntu is installed. Follow the official [Microsoft WSL guide](https://learn.microsoft.com/en-us/windows/wsl/install).


### **Install Docker**
Docker is required to run Kubernetes and containers. Install Docker Desktop for WSL2 if using WSL2 Ubuntu, follow the [official Docker WSL documentation](https://docs.docker.com/desktop/wsl/).

For native Ubuntu, install Docker using the following commands, follow the [official Docker installation guide](https://docs.docker.com/engine/install/ubuntu/).

### **Install kind**
Install kind (Kubernetes IN Docker) to create a local Kubernetes cluster:
```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.27.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

For more installation options and documentation, see [kind documentation](https://kind.sigs.k8s.io/docs/user/quick-start/).
### **Install kubectl**
Install **kubectl** to interact with Kubernetes clusters:
```bash
sudo apt-get update
# apt-transport-https may be a dummy package; if so, you can skip that package
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg

# If the folder `/etc/apt/keyrings` does not exist, it should be created before the curl command, read the note below.
# sudo mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.32/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
sudo chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg # allow unprivileged APT programs to read this keyring

# This overwrites any existing configuration in /etc/apt/sources.list.d/kubernetes.list
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.32/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo chmod 644 /etc/apt/sources.list.d/kubernetes.list   # helps tools such as command-not-found to work correctly

sudo apt-get update
sudo apt-get install -y kubectl
```

For further guidance, refer to [kubectl linux installation docs](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/).

### Install Helm

Install Helm to manage Kubernetes applications:

```bash
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
sudo apt-get install apt-transport-https --yes
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

For more information, see the [Helm installation guide](https://helm.sh/docs/intro/install/).

### Install lua, luarocks, and luasocket

Install lua, luarocks, and luasocket to run the AIOpsLab application:

Install Lua:
```bash
sudo apt install build-essential libreadline-dev unzip
curl -L -R -O http://www.lua.org/ftp/lua-5.3.5.tar.gz
tar zxf lua-5.3.5.tar.gz
cd lua-5.3.5
make linux test
sudo make install
```

Install LuaRocks and LuaSocket:
```bash
wget https://luarocks.org/releases/luarocks-3.11.1.tar.gz
tar zxpf luarocks-3.11.1.tar.gz
cd luarocks-3.11.1
./configure --with-lua-include=/usr/local/include && make && sudo make install
sudo luarocks install luasocket
```

For more information, see the [Lua installation guide](https://www.lua.org/download.html) and the [LuaRocks installation guide](https://github.com/luarocks/luarocks/blob/master/docs/index.md).

---

## Deployment Steps

### 1. Build the Custom KIND Image

The Dockerfile in this directory is designed specifically for Ubuntu running under WSL2 (amd64). **Please refer to this Dockerfile** to build an image that is compatible with your own machine —this is critical because the published AIOpsLab image
```jacksonarthurclark/aiopslab-kind:latest``` is built for macOS (arm64) and will not work on Ubuntu (amd64).

Build the custom image using:

```bash
docker build -t your_dockerhub_username/aiopslab-kind:latest -f kind/Dockerfile .
```
> **Note:** Replace `your_dockerhub_username` with your Docker Hub account if pushing the image.

### 2. (Optional) Push the Image to Dockerhub

If you wish to publish your custom image and have it referenced by the kind configuration file, push it to Docker Hub:

```bash
docker push your_dockerhub_username/aiopslab-kind:latest
```

Remember to update the `kind-config.yaml` file with your image name if you are using your own published image.

### 3. Create persistent directories for Prometheus data

Before proceeding the creation of the Kubernetes cluster, you need to ensure that the Prometheus data directory is created and has the correct permissions. This is necessary because the Prometheus container will write data to this directory. Run the following commands to create the directory and set the correct permissions:

```bash
sudo mkdir -p /mnt/datadrive/prometheus
sudo chown -R $(whoami):$(whoami) /mnt/datadrive/prometheus
sudo chmod -R 777 /mnt/datadrive/prometheus
```

### 4. Create a KIND Kubernetes Cluster

Use the provided or updated `kind-config.yaml` to create a Kubernetes cluster:

```bash
kind create cluster --config kind/kind-config.yaml
```

After finishing cluster creation, proceed to the next "Update config.yml" step.

---

## **Troubleshooting**

- **Docker Issues:**  
  Ensure Docker is running within your WSL2 environment. Verify with `docker ps` to list running containers.

- **Cluster Creation Failures:**  
  Check that Docker is correctly installed and that your system has enough resources (CPU, memory). Examine the output of `kind export logs <cluster-name>` for details.

- **Deployment Problems:**  
  Use `kubectl logs <pod-name>` to view pod logs and diagnose application issues. Make sure that your `kind-config.yaml` file references the correct image.

- **Resource Allocation:**  
  WSL2 may require additional resources. Adjust the WSL2 settings in your `.wslconfig` file on Windows if you encounter performance issues.

---

## **Conclusion**
This guide covers deploying **AIOpsLab** on **both WSL2 and Ubuntu 24.04**, ensuring compatibility across different environments. By following these steps, you can successfully set up **Docker, kind, and Kubernetes** and deploy the AIOpsLab application.

For advanced configurations, refer to the [AIOpsLab documentation](https://github.com/Flemington8/AIOpsLab). 🚀
