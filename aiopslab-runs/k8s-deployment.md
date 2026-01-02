# AIOpsLab Viewer - Kubernetes Deployment Guide

This guide covers deploying AIOpsLab Viewer to Kubernetes using Helm charts with persistent volumes for both database and runs data.

## 📋 Prerequisites

- Kubernetes cluster (1.19+)
- Helm 3.0+
- kubectl configured to access your cluster
- Docker (for building custom images)

## 🚀 Quick Start

### 1. Deploy to Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd aiopslab-runs

# Deploy using the deployment script
./scripts/k8s-deploy.sh

# Or deploy manually with Helm
helm install aiopslab-viewer ./helm/aiopslab-viewer \
  --namespace aiopslab \
  --create-namespace \
  --values ./helm/aiopslab-viewer/values-dev.yaml
```

### 2. Access the Application

```bash
# Port forward to access locally
kubectl port-forward -n aiopslab svc/aiopslab-viewer 3000:3000

# Visit http://localhost:3000
```

## 🏗️ Helm Chart Structure

```
helm/aiopslab-viewer/
├── Chart.yaml                 # Chart metadata
├── values.yaml               # Default configuration
├── values-dev.yaml           # Development overrides
├── values-prod.yaml          # Production overrides
└── templates/
    ├── _helpers.tpl          # Template helpers
    ├── deployment.yaml       # Main application deployment
    ├── service.yaml          # Kubernetes service
    ├── pvc.yaml             # Persistent Volume Claims
    ├── serviceaccount.yaml  # Service account
    ├── ingress.yaml         # Ingress configuration
    ├── hpa.yaml            # Horizontal Pod Autoscaler
    ├── configmap.yaml      # ConfigMap (optional)
    ├── secret.yaml         # Secrets (optional)
    └── networkpolicy.yaml  # Network policies (optional)
```

## 💾 Persistent Storage

The Helm chart creates two Persistent Volume Claims:

### Database PVC
- **Name**: `{release-name}-database`
- **Mount Path**: `/app/data`
- **Default Size**: 1Gi (dev) / 5Gi (prod)
- **Purpose**: Stores SQLite database (`runs.db`)

### Runs Data PVC
- **Name**: `{release-name}-runs`
- **Mount Path**: `/app/runs`
- **Default Size**: 10Gi (dev) / 50Gi (prod)
- **Purpose**: Stores AIOpsLab run data (logs, evaluations)

### Storage Class Configuration

```yaml
# values.yaml
persistence:
  database:
    enabled: true
    storageClass: "fast-ssd"  # Use your preferred storage class
    size: 5Gi
  runs:
    enabled: true
    storageClass: "fast-ssd"
    size: 50Gi
```

## 🔧 Configuration Options

### Environment Variables

All server configuration can be customized via Helm values:

```yaml
# values.yaml
env:
  NODE_ENV: production
  PORT: "3000"
  HTTPS_PORT: "3443"
  RUNS_PATH: "/app/runs"
  DATABASE_PATH: "/app/data/runs.db"
  RATE_LIMIT_WINDOW_MS: "900000"
  RATE_LIMIT_MAX_REQUESTS: "100"
  LOG_LEVEL: "info"
```

### Resource Limits

```yaml
# values.yaml
resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi
```

### Autoscaling

```yaml
# values.yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

## 🌐 Ingress Configuration

### Development (HTTP)

```yaml
# values-dev.yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: aiopslab-viewer.local
      paths:
        - path: /
          pathType: Prefix
```

### Production (HTTPS with TLS)

```yaml
# values-prod.yaml
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: aiopslab-viewer.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: aiopslab-viewer-tls
      hosts:
        - aiopslab-viewer.yourdomain.com
```

## 🔒 Security Features

### Pod Security Context

```yaml
podSecurityContext:
  fsGroup: 1000
  runAsNonRoot: true
  runAsUser: 1000

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: false
  runAsNonRoot: true
  runAsUser: 1000
```

### Network Policies

```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 3000
  egress:
    - to: []
      ports:
        - protocol: UDP
          port: 53  # DNS
```

## 📝 Deployment Commands

### Install New Deployment

```bash
# Development
helm install aiopslab-viewer ./helm/aiopslab-viewer \
  --namespace aiopslab \
  --create-namespace \
  --values ./helm/aiopslab-viewer/values-dev.yaml

# Production
helm install aiopslab-viewer ./helm/aiopslab-viewer \
  --namespace aiopslab-prod \
  --create-namespace \
  --values ./helm/aiopslab-viewer/values-prod.yaml \
  --set image.tag=v1.0.0
```

### Upgrade Existing Deployment

```bash
# Upgrade with new image
helm upgrade aiopslab-viewer ./helm/aiopslab-viewer \
  --namespace aiopslab \
  --values ./helm/aiopslab-viewer/values-dev.yaml \
  --set image.tag=v1.1.0

# Upgrade configuration only
helm upgrade aiopslab-viewer ./helm/aiopslab-viewer \
  --namespace aiopslab \
  --values ./helm/aiopslab-viewer/values-dev.yaml \
  --reuse-values
```

### Rollback Deployment

```bash
# View rollback history
helm history aiopslab-viewer -n aiopslab

# Rollback to previous version
helm rollback aiopslab-viewer -n aiopslab

# Rollback to specific revision
helm rollback aiopslab-viewer 2 -n aiopslab
```

### Uninstall Deployment

```bash
# Uninstall but keep PVCs
helm uninstall aiopslab-viewer -n aiopslab

# Delete PVCs manually if needed
kubectl delete pvc aiopslab-viewer-database -n aiopslab
kubectl delete pvc aiopslab-viewer-runs -n aiopslab
```

## 🔍 Monitoring and Debugging

### Check Pod Status

```bash
kubectl get pods -n aiopslab -l app.kubernetes.io/name=aiopslab-viewer
```

### View Logs

```bash
kubectl logs -n aiopslab -l app.kubernetes.io/name=aiopslab-viewer -f
```

### Check Storage

```bash
kubectl get pvc -n aiopslab
kubectl describe pvc aiopslab-viewer-database -n aiopslab
```

### Port Forward for Local Access

```bash
kubectl port-forward -n aiopslab svc/aiopslab-viewer 3000:3000
```

### Execute Commands in Pod

```bash
kubectl exec -it -n aiopslab deploy/aiopslab-viewer -- /bin/sh
```

## 🎛️ Customization Examples

### Custom Storage Classes

```yaml
# values-custom.yaml
persistence:
  database:
    storageClass: "ssd-retain"
    size: 10Gi
  runs:
    storageClass: "nfs-shared"
    size: 100Gi
```

### Multiple Environments

```bash
# Staging environment
helm install aiopslab-staging ./helm/aiopslab-viewer \
  --namespace aiopslab-staging \
  --create-namespace \
  --values ./helm/aiopslab-viewer/values-dev.yaml \
  --set env.NODE_ENV=staging \
  --set ingress.hosts[0].host=aiopslab-staging.company.com
```

### Custom SSL Certificates

```yaml
# values-custom-ssl.yaml
ssl:
  enabled: true
  existingSecret: "custom-ssl-cert"
```

```bash
# Create SSL secret
kubectl create secret tls custom-ssl-cert \
  --cert=path/to/cert.crt \
  --key=path/to/private.key \
  -n aiopslab
```

## 🔧 Troubleshooting

### Common Issues

#### 1. PVC Pending State

```bash
# Check storage class
kubectl get storageclass

# Check PVC events
kubectl describe pvc aiopslab-viewer-database -n aiopslab
```

#### 2. Pod CrashLoopBackOff

```bash
# Check pod logs
kubectl logs -n aiopslab -l app.kubernetes.io/name=aiopslab-viewer --previous

# Check pod description
kubectl describe pod -n aiopslab -l app.kubernetes.io/name=aiopslab-viewer
```

#### 3. Ingress Not Working

```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# Check ingress resource
kubectl describe ingress aiopslab-viewer -n aiopslab
```

### Performance Tuning

#### Resource Optimization

```yaml
# For high-traffic environments
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 60
```

#### Storage Performance

```yaml
# Use high-performance storage
persistence:
  database:
    storageClass: "premium-ssd"
  runs:
    storageClass: "premium-ssd"
```

## 📊 Production Considerations

### High Availability

```yaml
# Anti-affinity rules
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
            - key: app.kubernetes.io/name
              operator: In
              values:
                - aiopslab-viewer
        topologyKey: kubernetes.io/hostname
```

### Backup Strategy

```bash
# Backup database PVC
kubectl exec -n aiopslab deploy/aiopslab-viewer -- tar czf - /app/data | \
  kubectl exec -i backup-pod -- tar xzf - -C /backup/$(date +%Y%m%d)

# Backup runs data
kubectl exec -n aiopslab deploy/aiopslab-viewer -- tar czf - /app/runs | \
  kubectl exec -i backup-pod -- tar xzf - -C /backup/$(date +%Y%m%d)
```

### Monitoring Integration

```yaml
# Prometheus annotations
podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "3000"
  prometheus.io/path: "/metrics"
```

This comprehensive Kubernetes deployment setup provides enterprise-grade capabilities for running AIOpsLab Viewer in production environments with persistent data storage, security, and scalability features.
