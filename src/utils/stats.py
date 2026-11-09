class IncrementalMean:
    def __init__(self):
        self.sum: float = 0.0
        self.count: int = 0

    def add(self, value: float) -> None:
        self.count += 1
        self.sum += value

    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0