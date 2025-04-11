from aiopslab.service.apps.flower import Flower

class FlowerTest:
    def __init__(self):
        self.app = Flower()

    def start_workload(self):
        print("== Start Workload ==")
        self.app.deploy()