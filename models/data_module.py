import pytorch_lightning as pl
from torch.utils.data import  DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size):
        super().__init__()

        self.train_data = data[0]
        self.val_data = data[1]
        self.test_data = data[2]

        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.batch_size)