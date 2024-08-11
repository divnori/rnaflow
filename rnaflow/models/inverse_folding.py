import csv
import lightning.pytorch as pl
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append("geometric_rna_design/src")
from model import AutoregressiveMultiGNN
from rna_data import RNADesignDataset

from interpolant import Interpolant

def compute_perplexity(logits, targets):
    """
    Compute perplexity from logits and target sequence.
    """
    loss = F.cross_entropy(logits, targets.long(), reduction='mean')
    perplexity = torch.exp(loss)
    return perplexity

class InverseFoldingModel(pl.LightningModule):

    def __init__(self, smoothing=0):
        super(InverseFoldingModel, self).__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)

        self.model = AutoregressiveMultiGNN()
        self.data_featurizer = RNADesignDataset(device = 'cuda', num_conformers=1)

        self.as_denoiser = False

        self.csv_log_path = "scripts/lightning_logs/inverse_folding.csv"

        self.epoch_losses = []
        self.epoch_rna_aars = []

        self.val_epoch_losses = []
        self.val_epoch_rna_aars = []

    def prep_input(self, data, as_denoiser=False):

        if as_denoiser:
            interpolant = Interpolant()
            true_crds = torch.cat((data["prot_coords"][0], data["rna_coords"][0]),axis=0)[None,:]
            noise_mask = torch.cat((torch.zeros((1, data["prot_coords"].shape[1])), torch.ones((1, data["rna_coords"].shape[1]))), dim=1)
            noisy_rna_crds, timestep = interpolant.corrupt_coords(true_crds, noise_mask)

            rna_graph, rna_coords = self.data_featurizer._featurize(data["rna_seq"][0], noisy_rna_crds) # batch dim and conf dim match

            prot_graph, prot_coords = self.data_featurizer._featurize(data["prot_seq"][0], data["prot_coords"], rna=False)

            cplx_graph = self.data_featurizer._connect_graphs(prot_graph, rna_graph, data["prot_coords"], data["rna_coords"])
            return cplx_graph, prot_coords, rna_coords, timestep
        
        else:
            rna_graph, rna_coords = self.data_featurizer._featurize(data["rna_seq"][0], data["rna_coords"]) # batch dim and conf dim match
            prot_graph, prot_coords = self.data_featurizer._featurize(data["prot_seq"][0], data["prot_coords"], rna=False)
            cplx_graph = self.data_featurizer._connect_graphs(prot_graph, rna_graph, data["prot_coords"], data["rna_coords"])
            return cplx_graph, prot_coords, rna_coords, None

    def model_step(self, data, timestep=None, noisy=False):
        
        if not noisy: # already normalized in RNAFlow
            rna_centroid = torch.mean(data["rna_coords"], dim=(1,2))
            data["rna_coords"] -= rna_centroid
            data["prot_coords"] -= rna_centroid

        cplx_graph, prot_coords, rna_coords, timestep = self.prep_input(data, self.as_denoiser)
        logits = self.model(cplx_graph, timestep=timestep)
        true_seq = cplx_graph.seq

        is_rna = torch.cat((torch.zeros((data["prot_coords"].shape[1],)), torch.ones((data["rna_coords"].shape[1],))), dim=0).bool()
        logits_rna = logits[is_rna]

        one_hot = F.gumbel_softmax(logits_rna, tau=1, hard=True)
        pred_rna_detached = torch.nonzero(one_hot)[:,1].detach().cpu().numpy()

        pred_rna_seq = "".join([self.data_featurizer.rna_num_to_letter.get(x, "X") for x in pred_rna_detached.tolist()])

        true_rna_seq = true_seq[is_rna,0]
        loss_value = self.loss_fn(logits_rna, true_rna_seq)

        rna_correct = (pred_rna_detached == true_rna_seq.detach().cpu().numpy()).sum()
        rna_recovery_rate = rna_correct/logits_rna.shape[0]        
        return loss_value, rna_recovery_rate, pred_rna_seq, one_hot

    def training_step(self, batch):
        pdb_id, data = batch
        pdb_id = pdb_id[0]
        loss_value, rna_recovery_rate, pred_rna_seq, one_hot = self.model_step(data)
        outputs = {"loss": loss_value, "rna_aar": rna_recovery_rate}
        self.epoch_losses.append(outputs["loss"])
        self.epoch_rna_aars.append(outputs["rna_aar"])
        return outputs["loss"]
    
    def validation_step(self, batch):
        pdb_id, data = batch
        pdb_id = pdb_id[0]
        loss_value, rna_recovery_rate, pred_rna_seq, one_hot = self.model_step(data)
        outputs = {"val_loss": loss_value, "rna_aar": rna_recovery_rate}
        self.val_epoch_losses.append(outputs["val_loss"])
        self.val_epoch_rna_aars.append(outputs["rna_aar"])
        self.log("val_loss", loss_value)

    def predict_step(self, batch, data=None, timestep=None, in_rnaflow=False):

        for i in range(1): # sampling
        
            if not in_rnaflow:
                pdb_id, data = batch
                pdb_id = pdb_id[0]

                rna_centroid = torch.mean(data["rna_coords"], dim=(1,2))
                data["prot_coords"] -= rna_centroid
                data["rna_coords"] -= rna_centroid
                
            n_samples = 1

            cplx_graph, prot_coords, rna_coords, timestep = self.prep_input(data)
            true_seq = cplx_graph.seq

            is_rna = torch.cat((torch.zeros((data["prot_coords"].shape[1],)), torch.ones((data["rna_coords"].shape[1],))), dim=0).bool()

            samples, logits = self.model.sample(cplx_graph, n_samples, timestep=timestep, temperature=0.1, is_rna_mask=is_rna) # autoregressive
            pred_rna = samples[0,is_rna]
            logits_rna = logits[is_rna]
            rank_perplexity = compute_perplexity(logits_rna, pred_rna)
            true_rna_seq = true_seq[is_rna.bool(),0]
            eval_perplexity = compute_perplexity(logits_rna, true_rna_seq)
        
            rna_correct = (pred_rna.cpu().detach().numpy() == true_rna_seq.cpu().detach().numpy()).sum()
            rna_recovery_rate = rna_correct/true_rna_seq.shape[0]

            pred_rna_seq = "".join([self.data_featurizer.rna_num_to_letter.get(x, "X") for x in pred_rna.tolist()])
            
            if not in_rnaflow:
                outputs = {"rna_recovery": rna_recovery_rate, "pdb_ids": pdb_id, "pred_seqs": pred_rna_seq, "eval_perplexity": eval_perplexity, "rank_perplexity": rank_perplexity}
                self.log_to_csv(outputs)
            else:
                pred_one_hot = F.one_hot(pred_rna.long(), num_classes=4)
                return None, rna_recovery_rate, pred_rna_seq, pred_one_hot, eval_perplexity, rank_perplexity

    def design_rna(self, prot_seq, prot_coords, rna_seq, rna_coords, timestep):
        is_rna = torch.cat((torch.zeros((len(prot_seq),)), torch.ones((rna_coords.shape[0],))), dim=0).bool()
        rna_graph, rna_coords = self.data_featurizer._featurize(rna_seq, rna_coords.unsqueeze(0)) # batch dim and conf dim match
        prot_graph, prot_coords = self.data_featurizer._featurize(prot_seq, prot_coords.unsqueeze(0), rna=False)
        cplx_graph = self.data_featurizer._connect_graphs(prot_graph, rna_graph, prot_coords, rna_coords)
        samples, logits = self.model.sample(cplx_graph, 1, timestep, temperature=0.1, is_rna_mask=is_rna) # autoregressive
        pred_rna = samples[0,is_rna]
        pred_rna_seq = "".join([self.data_featurizer.rna_num_to_letter.get(x, "X") for x in pred_rna.tolist()])
        pred_one_hot = F.one_hot(pred_rna.long(), num_classes=4)
        return pred_rna_seq, pred_one_hot

    
    def on_train_epoch_end(self):
        avg_loss = sum(self.epoch_losses)/len(self.epoch_losses)
        avg_rna_aar = sum(self.epoch_rna_aars)/len(self.epoch_rna_aars)

        logs = {"epoch": self.current_epoch, "avg_loss": avg_loss.item(), "avg_rna_aar": avg_rna_aar.item()}
        self.log_to_csv(logs)

        self.epoch_losses = []
        self.epoch_rna_aars = []

    def on_val_epoch_end(self):
        avg_loss = sum(self.val_epoch_losses)/len(self.val_epoch_losses)
        avg_rna_aar = sum(self.val_epoch_rna_aars)/len(self.val_epoch_rna_aars)

        logs = {"val_epoch": self.current_epoch, "val_avg_loss": avg_loss.item(), "val_avg_rna_aar": avg_rna_aar.item()}
        self.log_to_csv(logs)

        self.val_epoch_losses = []
        self.val_epoch_rna_aars = []

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=0.001        
        )
    
    def log_to_csv(self, outputs):
        fieldnames = ["epoch"] + list(outputs.keys())

        # Check if the CSV file exists, create it if not
        write_header = not os.path.exists(self.csv_log_path)
        with open(self.csv_log_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if the file is newly created
            if write_header:
                writer.writeheader()

            # Write values for the current step
            row = {"epoch": self.trainer.current_epoch}
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    row[key] = value.item()
                else:
                    row[key] = value
            writer.writerow(row)