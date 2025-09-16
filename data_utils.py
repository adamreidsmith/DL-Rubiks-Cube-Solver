class RunningMean:
    def __init__(self):
        self.sum: float = 0.0
        self.n_points: int = 0

    def update(self, val):
        self.sum += val
        self.n_points += 1

    def get(self):
        return self.sum / self.n_points

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return self.__str__()


class RunningMin:
    def __init__(self):
        self.val = float('inf')

    def update(self, val):
        if val < self.val:
            self.val = val

    def get(self):
        return self.val

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return self.__str__()
