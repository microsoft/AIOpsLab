## TiDB Cluster and Operator Deployment with Ansible

### Set up local PVs for TiDB cluster

```shell
ansible-playbook -i ../cloudlab_inventory.yml tidb_pv_setup.yml 
```

This will set up the local PVs for TiDB cluster to use in the self-managed Kubernetes cluster.
It will create a loop device with four directories mounted on it under `agent-ops/tidb/`.

### Deploy TiDB Operator and Cluster

```shell
ansible-playbook -i ../cloudlab_inventory.yml tidb_operator_cluster.yml 
```

This will deploy the TiDB Operator and a TiDB cluster with dashboard and monitor.

### Check the Deployment Status

Check the status of the TiDB Operator:
```shell
kubectl get pods --namespace tidb-admin -l app.kubernetes.io/instance=tidb-operator
```

Check the status of the TiDB cluster:
```shell
kubectl get po -n tidb-cluster
```

We should also consider the local Kind cluster setup, which is much easier (without the need to setup local PV and bind the storageclass): https://docs.pingcap.com/tidb-in-kubernetes/stable/get-started#step-2-deploy-tidb-operator
