class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input, n=1.0):
        self.count += n
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                if k not in self.data:
                    self.data[k] = v
                else:
                    self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}

    def reset(self):
        self.data = {}
        self.count = 0
