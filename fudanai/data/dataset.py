class Dataset:
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
