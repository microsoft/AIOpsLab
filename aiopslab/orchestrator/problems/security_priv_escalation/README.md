# Security benchmark: Privileged-pod compromise (Falco + OPA)

A security-oriented AIOpsLab problem for agents that can reason about **both**
operations and security. It is the canonical Falco/OPA demo turned into a
benchmark task.

## Scenario

The `geo` workload of **HotelReservation** is patched to run **privileged**
(`privileged: true`, `runAsUser: 0`) with the **node root filesystem mounted**
(`hostPath: /`). A short **runtime intrusion** is then executed inside the pod:
an interactive shell reads `/etc/shadow` and touches the host mount.

The application keeps serving traffic — health, latency, and error telemetry all
look normal. The incident is only visible through **two independent security
signals** that the agent must correlate:

| Signal | Tool | How to see it |
|--------|------|---------------|
| Privileged container / hostPath mount (static config) | **OPA Gatekeeper** (audit / `dryrun`) | `kubectl get k8spspprivilegedcontainer.constraints.gatekeeper.sh -o yaml` |
| Shell-in-container + sensitive-file read (runtime) | **Falco** | `kubectl logs -n falco -l app.kubernetes.io/name=falco -c falco` |

Because neither signal alone is conclusive, there is no one-shot answer — the
agent has to combine a configuration finding with a runtime alert to conclude
the `geo` workload is compromised.

## Tasks (variants)

| Problem ID | Task | Expected answer / success |
|------------|------|---------------------------|
| `security_priv_escalation_hotel_res-detection-1` | Detection | `submit("Yes")` |
| `security_priv_escalation_hotel_res-localization-1` | Localization | `submit(["geo"])` |
| `security_priv_escalation_hotel_res-mitigation-1` | Mitigation | remove privileged + hostPath posture **and** keep pods healthy |

The mitigation eval is intentionally stricter than the default "pods are
Running" check (the compromised pod is Running the whole time): success requires
`security_posture_fixed AND app_healthy`.

## What the harness deploys automatically

`init_problem()` → deploys HotelReservation → `inject_fault()`:

1. Installs **Falco** (Helm, `falco` ns, `modern_ebpf` driver) and **OPA
   Gatekeeper** (Helm, `gatekeeper-system` ns) — see
   [`aiopslab/service/security_tooling.py`](../../../service/security_tooling.py).
   Gatekeeper constraints are loaded in **audit mode** so the insecure pod is
   admitted (not blocked).
2. Patches `geo` to the privileged/host-mounted posture and runs the intrusion —
   see [`aiopslab/generators/fault/inject_security.py`](../../../generators/fault/inject_security.py).

Falco is installed **before** the intrusion so it captures the runtime event.

## Running it

```bash
python3 cli.py
(aiopslab) $ start security_priv_escalation_hotel_res-detection-1
```

## Requirements / caveats

- The host running AIOpsLab needs `helm` and `kubectl` on PATH with a working
  kubeconfig (same assumption as the other fault injectors).
- Falco uses the `modern_ebpf` driver (no kernel headers needed), which works on
  kind and most modern VMs. On an unusual kernel you may need to switch the
  driver in `security_tooling.py`.
- First injection is slow (~minutes) because it `helm install ... --wait`s Falco
  and Gatekeeper.
- This is a POC / first canonical security case; the `geo` service is
  parameterizable if you want to add variants for other workloads.
