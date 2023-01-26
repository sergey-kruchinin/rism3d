class Rism3DMaxStepError(Exception):
    def __init__(self, message, step=None, accuracy=None):
        self.message = message
        self.step = step
        self.accuracy = accuracy

class Rism3DConvergenceError(Exception):
    def __init__(self, message, step=None):
        self.message = message
        self.step = step
