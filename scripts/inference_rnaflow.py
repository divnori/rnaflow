from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import sys

sys.path.append("rnaflow")
from data.dataloader import RFDataModule
from models.rnaflow import RNAFlow
from models.inverse_folding import InverseFoldingModel

RF_DATA_FOLDER = "rnaflow/data/rf_data"
DATASET_PKL = "rnaflow/data/rf2na_dataset.pickle"
split = "rf2na"

if __name__ == "__main__":
    print("Running RNAFlow Inference.")

    data_module = RFDataModule(rf_data_folder=RF_DATA_FOLDER, dataset_pkl=DATASET_PKL, batch_size=1)
    test_dataloader = data_module.test_dataloader()

    rnaflow = RNAFlow()
    rnaflow.denoise_model = InverseFoldingModel.load_from_checkpoint("checkpoints/rf2na_pretrained_inverse_folder.ckpt")

    trainer = Trainer(devices=1)
    trainer.predict(rnaflow, dataloaders=test_dataloader)