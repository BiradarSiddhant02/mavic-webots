from controller import Supervisor   # type: ignore

class Environment(Supervisor):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.simulationReset()
        self.simulationResetPhysics()