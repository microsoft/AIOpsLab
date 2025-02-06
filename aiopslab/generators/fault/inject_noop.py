from aiopslab.service.kubectl import KubeCtl
from aiopslab.generators.fault.base import FaultInjector


class NoopFaultInjector(FaultInjector):
    def __init__(self, namespace: str):
        super().__init__(namespace)
        self.namespace = namespace
        self.kubectl = KubeCtl()

    def inject_no_op(self, _, __):
        pass

    def recover_no_op(self):
        pass


if __name__ == "__main__":
    namespace = "test-hotel-reservation"
    microservices = ["geo"]
    fault_type = "no_op"
    injector = NoopFaultInjector(namespace)
    injector._inject(fault_type, microservices, "30s")
    injector._recover(fault_type)
