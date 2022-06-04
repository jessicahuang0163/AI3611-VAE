class RunningStats:
    def __init__(self):
        self.n_old = 0
        self.n = 0
        self.old_m = 0
        self.new_m = 0

    def clear(self):
        self.n_old = 0
        self.n = 0

    def push(self, x, num=1):
        self.n += num
        if self.n == 1:
            self.old_m = self.new_m = x
        else:
            self.new_m = (self.old_m * self.n_old + x * num) / self.n
            self.old_m = self.new_m
        self.n_old = self.n

    def mean(self):
        return self.new_m if self.n else 0.0
