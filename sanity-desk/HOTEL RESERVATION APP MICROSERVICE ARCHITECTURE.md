# Super MVP — sc-mvp/app-deploy

## 1️⃣ Goal
Deploy a **hotel reservation application** onto the SC-MVP Kubernetes cluster and validate that the deployment is **idempotent, observable, and verifiably successful**, forming **Checkpoint SC-MVP-02** in the pipeline.

---

## 2️⃣ Requirements

| Requirement | Details |
| ----------- | ------- |
| **Target Cluster** | Existing SC-MVP cluster (from SC-MVP-01) |
| **Namespace** | `sc-mvp-test` |
| **Deployment Method** | Declarative Kubernetes manifests |
| **Idempotency** | Re-applying manifests must not cause errors or drift |
| **Verification Tools** | kubectl |
| **Logging Format** | CSV / JSON (SQLite ETL compatible) |
| **Rollback Strategy** | Automatic or manual rollback on failure |

---

## 3️⃣ Deployment Steps

### 3.1 Application Manifests
- Kubernetes **Deployment** manifest for the hotel reservation app
- Kubernetes **Service** manifest for internal/external access
- Optional **Ingress** manifest (environment-dependent)

---

### 3.2 Apply Deployment
- Apply manifests to the `sc-mvp-test` namespace
- Ensure repeated `kubectl apply` operations are safe and consistent

---

### 3.3 Rollout Verification
- Validate rollout completion using:
  - `kubectl rollout status`
  - Pod readiness and availability checks
- Confirm service endpoint accessibility

---

### 3.4 Idempotency Validation
- Re-apply manifests
- Confirm:
  - No unintended restarts
  - No configuration drift
  - Stable pod state

---

## 4️⃣ Logging & Verification
- Capture stdout/stderr from deployment commands
- Record:
  - Deployment timestamps (apply / start / ready)
  - Pod status and restart counts
  - Image tag and/or digest used
- Store logs in **CSV/JSON** format for SQLite ingestion
- Mark checkpoint as **PASS** only if rollout completes successfully

---

## 5️⃣ Failure Handling & Safety
- On failure:
  - Trigger rollback or enter explicit failed state
  - Preserve cluster state and logs for audit
- Do not auto-delete resources unless explicitly configured

---

## 6️⃣ Deliverables
- Application deployment manifests
- Rollout verification logs (CSV/JSON)
- Idempotency validation evidence
- Documentation describing:
  - Deployment procedure
  - Verification steps
  - Re-run behavior and constraints

---

## Checkpoint Summary

**Checkpoint ID:** SC-MVP-02  
**Branch:** `sc-mvp/app-deploy`  
**Purpose:** App Deployment & Rollout Verification  
**Status:** ☐ PASS / ☐ FAIL (to be recorded during execution)
