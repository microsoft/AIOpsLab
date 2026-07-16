# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Install and manage in-cluster security tooling: Falco + OPA Gatekeeper.

Security-oriented benchmark problems install this tooling so that a
security-capable agent's arsenal has live data to query:

  - Falco       -> runtime (syscall/eBPF) detection. Emits alerts to the
                   stdout of the falco pods, readable via `kubectl logs`.
  - Gatekeeper  -> admission/audit policy detection (OPA). Records policy
                   violations on the running workloads.

The Gatekeeper constraints are installed in *audit* mode
(`enforcementAction: dryrun`) on purpose: an insecure workload must still be
admitted and stay healthy so that the agent has to actively discover it,
rather than having the platform block the deployment outright.

NOTE: This helper shells out to `helm` and `kubectl` on the machine running
AIOpsLab (the same way the other fault injectors do). The host must therefore
have `helm` and `kubectl` on its PATH and a working kubeconfig.
"""

import time

from aiopslab.service.kubectl import KubeCtl

FALCO_NAMESPACE = "falco"
GATEKEEPER_NAMESPACE = "gatekeeper-system"

FALCO_REPO_NAME = "falcosecurity"
FALCO_REPO_URL = "https://falcosecurity.github.io/charts"
GATEKEEPER_REPO_NAME = "gatekeeper"
GATEKEEPER_REPO_URL = "https://open-policy-agent.github.io/gatekeeper/charts"

# --- Gatekeeper policy assets --------------------------------------------------

# ConstraintTemplate: forbid privileged containers.
PRIVILEGED_TEMPLATE = """
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8spspprivilegedcontainer
spec:
  crd:
    spec:
      names:
        kind: K8sPSPPrivilegedContainer
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8spspprivileged
        violation[{"msg": msg}] {
          c := input_containers[_]
          c.securityContext.privileged
          msg := sprintf("Privileged container is not allowed: %v", [c.name])
        }
        input_containers[c] {
          c := input.review.object.spec.containers[_]
        }
        input_containers[c] {
          c := input.review.object.spec.initContainers[_]
        }
"""

# ConstraintTemplate: forbid hostPath volumes.
HOSTPATH_TEMPLATE = """
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8spsphostfilesystem
spec:
  crd:
    spec:
      names:
        kind: K8sPSPHostFilesystem
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8spsphostfilesystem
        violation[{"msg": msg}] {
          volume := input.review.object.spec.volumes[_]
          volume.hostPath
          msg := sprintf("HostPath volume is not allowed: %v", [volume.name])
        }
"""

# Constraints are rendered per-namespace and applied in dryrun (audit) mode.
PRIVILEGED_CONSTRAINT = """
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sPSPPrivilegedContainer
metadata:
  name: psp-privileged-container
spec:
  enforcementAction: dryrun
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces: ["{namespace}"]
"""

HOSTPATH_CONSTRAINT = """
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sPSPHostFilesystem
metadata:
  name: psp-host-filesystem
spec:
  enforcementAction: dryrun
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces: ["{namespace}"]
"""


class SecurityTooling:
    """Deploy/tear down Falco and OPA Gatekeeper for security benchmarks."""

    def __init__(self):
        self.kubectl = KubeCtl()

    # --- public API ------------------------------------------------------------

    def deploy_all(self, app_namespace: str):
        """Install Falco and Gatekeeper and load audit policies for `app_namespace`."""
        self.deploy_falco()
        self.deploy_gatekeeper()
        self.load_policies(app_namespace)

    def cleanup(self):
        """Best-effort removal of the security tooling and its policies."""
        print("== Security Tooling Cleanup ==")
        self._apply_yaml(PRIVILEGED_CONSTRAINT.format(namespace="_"), delete=True)
        self._apply_yaml(HOSTPATH_CONSTRAINT.format(namespace="_"), delete=True)
        self._apply_yaml(PRIVILEGED_TEMPLATE, delete=True)
        self._apply_yaml(HOSTPATH_TEMPLATE, delete=True)
        self._run(f"helm uninstall gatekeeper -n {GATEKEEPER_NAMESPACE}")
        self._run(f"helm uninstall falco -n {FALCO_NAMESPACE}")

    # --- Falco -----------------------------------------------------------------

    def deploy_falco(self):
        print("== Deploying Falco ==")
        if self._helm_release_exists("falco", FALCO_NAMESPACE):
            print("Falco already installed. Skipping.")
            return
        self._add_repo(FALCO_REPO_NAME, FALCO_REPO_URL)
        # modern_ebpf needs no kernel headers, so it works on kind and most VMs.
        # json_output makes the alerts easy to parse; falcosidekick is unneeded
        # because the agent reads alerts straight from the pod logs.
        cmd = (
            f"helm upgrade --install falco {FALCO_REPO_NAME}/falco "
            f"--namespace {FALCO_NAMESPACE} --create-namespace "
            "--set tty=true "
            "--set driver.kind=modern_ebpf "
            "--set collectors.kubernetes.enabled=true "
            "--set falcosidekick.enabled=false "
            "--set falco.json_output=true "
            "--wait --timeout 6m"
        )
        print(self._run(cmd))

    # --- Gatekeeper ------------------------------------------------------------

    def deploy_gatekeeper(self):
        print("== Deploying OPA Gatekeeper ==")
        if not self._helm_release_exists("gatekeeper", GATEKEEPER_NAMESPACE):
            self._add_repo(GATEKEEPER_REPO_NAME, GATEKEEPER_REPO_URL)
            cmd = (
                f"helm upgrade --install gatekeeper {GATEKEEPER_REPO_NAME}/gatekeeper "
                f"--namespace {GATEKEEPER_NAMESPACE} --create-namespace "
                "--set replicas=1 "
                "--set audit.interval=30 "
                "--wait --timeout 6m"
            )
            print(self._run(cmd))
        else:
            print("Gatekeeper already installed. Skipping.")

    def load_policies(self, app_namespace: str):
        """Apply constraint templates + (dryrun) constraints scoped to a namespace."""
        print("== Loading Gatekeeper audit policies ==")
        self._apply_yaml(PRIVILEGED_TEMPLATE)
        self._apply_yaml(HOSTPATH_TEMPLATE)

        # The constraint CRDs only exist once Gatekeeper reconciles the templates.
        self._wait_for_crd("k8spspprivilegedcontainer.constraints.gatekeeper.sh")
        self._wait_for_crd("k8spsphostfilesystem.constraints.gatekeeper.sh")

        self._apply_yaml(PRIVILEGED_CONSTRAINT.format(namespace=app_namespace))
        self._apply_yaml(HOSTPATH_CONSTRAINT.format(namespace=app_namespace))

    # --- helpers ---------------------------------------------------------------

    def _run(self, command: str) -> str:
        return self.kubectl.exec_command(command)

    def _add_repo(self, name: str, url: str):
        self._run(f"helm repo add {name} {url}")
        self._run("helm repo update")

    def _helm_release_exists(self, release: str, namespace: str) -> bool:
        out = self._run(f"helm status {release} -n {namespace}")
        return "STATUS: deployed" in out

    def _apply_yaml(self, manifest: str, delete: bool = False):
        verb = "delete --ignore-not-found" if delete else "apply"
        return self.kubectl.exec_command(f"kubectl {verb} -f -", input_data=manifest)

    def _wait_for_crd(self, crd_name: str, max_wait: int = 90, sleep: int = 3):
        waited = 0
        while waited < max_wait:
            out = self._run(f"kubectl get crd {crd_name} --no-headers")
            if crd_name in out:
                return
            time.sleep(sleep)
            waited += sleep
        print(f"[WARN] Timed out waiting for CRD {crd_name} to be established.")
