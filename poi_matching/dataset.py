from torch.utils.data import Dataset


class POIRecommendDataset(Dataset):
    def __init__(self, X, y, categories):
        self.X = X
        self.y = y
        self.categories = categories

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.categories[idx]
