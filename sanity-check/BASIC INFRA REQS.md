# Super MVP — sc-mvp/cluster-setup

## 1️⃣ Goal
Spin up a **lightweight, reproducible Kubernetes cluster** using **k3s or kind**, forming the foundation for the SC-MVP pipeline.

---

## 2️⃣ Requirements

| Requirement | Details |
| ----------- | ------- |
| **Cluster Type** | k3s (lightweight Kubernetes) or kind (Kubernetes in Docker) |
| **Nodes** | 1 control-plane + 1 worker node minimum |
| **Resource Limits** | CPU: 500m, Memory: 512Mi per node |
| **Namespace Isolation** | Optional namespace `sc-mvp-test` for app deployments |
| **Tools Required** | Docker, kubectl, kind or k3s CLI |
| **Reproducibility** | Cluster should be reproducible on any contributor machine |

---

## 3️⃣ Deployment Steps (Kind Example)

1. **Create a kind configuration file**

`kind-config.yaml`:

2. **Create the cluster**

`kind create cluster --name sc-mvp --config kind-config.yaml`

3. **Verify cluster status**

`
kubectl cluster-info --context kind-sc-mvp
kubectl get nodes
`

4. **Opt: Create namespace for testing**

`kubectl create namespace sc-mvp-test`

5. **Check namespace**

`kubectl get ns`

4️⃣ Logging & Verification

1. Capture stdout/stderr of `kind create cluster`.

2. Record `kubectl get nodes` output.

3. Timestamp logs for reproducibility.

4. Ensure nodes are `Ready` before proceeding to app deployment.

5️⃣ Optional Safety Features

1. Delete cluster on failure or cleanup:

2. `kind delete cluster --name sc-mvp`

3. Set resource constraints in Docker for nodes if machine resources are limited.

4. Verify Docker daemon is running before cluster creation.

6️⃣ Deliverables

1. `kind-config.yaml` → Cluster configuration manifest

2. Deployment logs → `sc-mvp-cluster-setup.log`

3. Optional namespace `sc-mvp-test` for app deployment

4. Documentation of steps for reproducibility
```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
