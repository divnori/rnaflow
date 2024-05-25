import pickle
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class RFDataModule(LightningDataModule):

    def __init__(self, rf_data_folder, dataset_pkl, batch_size):
        super().__init__()

        with open(dataset_pkl, 'rb') as handle:
            dataset = pickle.load(handle)
            train_dict = dataset["train"]
            test_dict = dataset["test"]
            val_dict = dataset["val"]
            
        self.data_dir = rf_data_folder
        self.test_dict = test_dict
        
        self.train_set = list(train_dict.items())
        self.val_set = list(val_dict.items())
        self.test_set = list(test_dict.items())
        self.batch_size = batch_size

        self.num_train = len(train_dict)
        self.num_val = len(val_dict)
        self.num_test = len(test_dict)

        print(f"Train Set: {len(self.train_set)}")
        print(f"Val Set: {len(self.val_set)}")
        print(f"Test Set: {len(self.test_set)}")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=1)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1)
    
    def test_by_id(self, pdb_id):
        return self.test_dict[pdb_id]