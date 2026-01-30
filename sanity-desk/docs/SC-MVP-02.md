---
title: "SC-MVP-02: Hotel Reservation App Deployment & Rollout"
branch: "sc-mvp/app-deploy"
commit: "Define app deployment checkpoint with idempotent rollout verification"
author: "vitrixlabph@gmail.com"
status: "Code and manifests included"
---

# SC-MVP-02: Hotel Reservation App Deployment & Rollout Documentation

## 1. Purpose

This document captures the **planning, checkpoint, and implementation** of SC-MVP-02:

- Deploy the **Hotel Reservation App** into the `sc-mvp-test` namespace
- Ensure **idempotent deployment** (safe re-apply with no side effects)
- Include both **documentation and code** for review and mapping

---

## 2. Components

### 2.1 Checkpoint Documentation
- Defines **deployment goals** and **rollout verification steps**
- Includes **metadata capture**, logging, and failure handling
- Serves as **control contract** for automation pipelines

### 2.2 Application Code
- Language: **Node.js + Express**
- Functionality:
  - `GET /health` → Kubernetes liveness/readiness probe
  - `GET /reservations` → List reservations (in-memory)
  - `POST /reservations` → Add reservation (in-memory)
- Location: `app/` folder
- Dockerized via `Dockerfile` (port 8080)

### 2.3 Kubernetes Manifests
- Location: `k8s/` folder
- Files:
  - `deployment.yaml` → Deployment configuration
  - `service.yaml` → Service configuration
- Features:
  - Rolling update strategy
  - Probes matching `/health` endpoint
  - Idempotent, replayable deployment

---

## 3. Deployment Instructions

**Local/Kind Testing Steps:**

```bash
# 1. Ensure kind is installed
kind version

# 2. Create cluster
kind create cluster --name sc-mvp

# 3. Create namespace
kubectl create namespace sc-mvp-test

# 4. Build and load Docker image
docker build -t hotel-reservation-app:mvp ./app
kind load docker-image hotel-reservation-app:mvp --name sc-mvp

# 5. Apply Kubernetes manifests
kubectl apply -f k8s/

# 6. Verify rollout
kubectl rollout status deployment/hotel-reservation-app -n sc-mvp-test

# 7. Test service (port-forward)
kubectl port-forward svc/hotel-reservation-app 8080:80 -n sc-mvp-test
curl http://localhost:8080/health
```
## 4. Rollout Verification
 
- **kubectl rollout status confirms deployment**

- **Pod readiness and availability verified via probes**

- **Service accessibility confirmed via port-forward**

- **Logs and metadata captured for audit:**

- - Pod status & restart count

- - Image digest / tag

- - Timestamps (apply/start/ready)

- **Failure handling:**

- - Automatic rollback or explicit failure state

- - Logs preserved for auditing

## 5. Artifact Integrity

**Artifact:	SHA256**
- Original: f465ef5a9022bb7d10c7bf324e816192c20493d5
- Microsoft Official:	<SHA256-MICROSOFT-PLACEHOLDER>

## 6. Notes

- Both documentation and code are included in this PR

- Keeps planning, checkpoint, and execution in a single reviewable bundle

- Supports future automation via SC-MVP-02 checkpoint YAML and GitHub Actions








