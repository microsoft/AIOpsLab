from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.base import Application


def get_frontend_url(app: Application):
    kubectl = KubeCtl()
    endpoint = kubectl.get_cluster_ip(app.frontend_service, app.namespace)
    return f"http://{endpoint}:{app.frontend_port}"
