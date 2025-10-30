# Building your own image

This document provides detailed, step-by-step instructions for building your own kind image to deploy AIOpsLab.

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
- [Deployment Steps](#deployment-steps)
  - [1. Build the Custom KIND Image](#1-build-the-custom-kind-image)
  - [2. (Optional) Push the Image to Dockerhub](#2-optional-push-the-image-to-dockerhub)
  - [3. Create a kind Kubernetes Cluster](#3-create-a-kind-kubernetes-cluster)
- [Troubleshooting](#troubleshooting)
- [Conclusion](#conclusion)

---

## **Overview**
AIOpsLab is deployed using **containerized components** and Kubernetes manifests. This guide provides a step-by-step deployment process, covering:

- Setting up **WSL2 (Windows Subsystem for Linux) or native Ubuntu 24.04**.
- Installing **Docker, kind, kubectl, Helm, Lua, Luarocks, and Luasocket**.
- Building a custom **kind image** and deploying AIOpsLab into a **local Kubernetes cluster**.

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
Install kind (Kubernetes IN Docker) to create a local Kubernetes cluster (for the latest release, [kind releases](https://github.com/kubernetes-sigs/kind/releases)):
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
sudo apt-get install curl gpg apt-transport-https --yes
curl -fsSL https://packages.buildkite.com/helm-linux/helm-debian/gpgkey | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/helm.gpg] https://packages.buildkite.com/helm-linux/helm-debian/any/ any main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

For more information, see the [Helm installation guide](https://helm.sh/docs/intro/install/).

---

## Deployment Steps

### 1. Build the Custom KIND Image

The Dockerfile in this directory is designed specifically for Ubuntu running under WSL2 (amd64). **Please refer to this [Dockerfile](/kind/Dockerfile)** to build an image that is compatible with your own machine 

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

Remember to create a `kind-config-custom.yaml` file similar to [kind-config-x86.yaml](/kind/kind-config-x86.yaml) with your image name if you are using your own published image.

### 3. Create a kind Kubernetes Cluster
```
# Using custom image
kind create cluster --config kind/kind-config-custom.yaml
```

After finishing cluster creation, proceed to the next "Update config.yml" step.


### 3. (Optional) Pre-pull and Load Required Images

If you experience slow image downloads during cluster setup, you can pre-pull the required images and then load them into your kind cluster:

1. First, pre-pull all required images from their registries:

    ```bash
    ./kind/pull_images.sh
    ```
  
    This script reads from images.txt and downloads all required images to your local Docker cache.

2. After creating your kind cluster, load the pre-pulled images into it:

    ```bash
    ./kind/load_images.sh [cluster-name]
    ```
    
    Replace `[cluster-name]` with your kind cluster name (optional - uses default cluster if not specified).

This approach will reduce deployment time by eliminating the need to download images during pod creation.

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
