from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import sys

sys.path.append("rnaflow")
from data.dataloader import RFDataModule
from models.rnaflow import RNAFlow
from models.inverse_folding import InverseFoldingModel

RF_DATA_FOLDER = "rnaflow/data/rf_data"
DATASET_PKL = "rnaflow/data/seq_sim_dataset.pickle"

if __name__ == "__main__":
    print("Running RNAFlow Inference.")

    data_module = RFDataModule(rf_data_folder=RF_DATA_FOLDER, dataset_pkl=DATASET_PKL, batch_size=1)
    test_dataloader = data_module.test_dataloader()

    rnaflow = RNAFlow.load_from_checkpoint("checkpoints/seq-sim-rnaflow-epoch32.ckpt")

    trainer = Trainer(devices=1)
    trainer.predict(rnaflow, dataloaders=test_dataloader)