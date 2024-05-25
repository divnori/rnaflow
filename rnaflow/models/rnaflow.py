import csv
import lightning.pytorch as pl
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

network_path = os.path.join('RoseTTAFold2NA/network')
sys.path.append(network_path)

data_path = os.path.join('rnaflow/data')
sys.path.append(data_path)

from predict import Predictor
import frame_utils
import pdb_utils
from inverse_folding import InverseFoldingModel
from interpolant import Interpolant, _centered_gaussian, NM_TO_ANG_SCALE

model = "RoseTTAFold2NA/network/weights/RF2NA_apr23.pt"
RF_DATA_FOLDER = "rf_data"

class RNAFlow(pl.LightningModule):

    def __init__(self):
        super(RNAFlow, self).__init__()

        if (torch.cuda.is_available()):
            self.folding_model = Predictor(model, torch.device("cuda"))
        else:
            self.folding_model = Predictor(model, torch.device("cpu"))

        self.csv_log_path = "/home/dnori/rna-design/src/scripts/lightning_logs/rnaflow.csv"

        self.mse_loss = nn.MSELoss()
        self.pyrimidine_indices_to_set_1 = torch.tensor([1, 5, 12])
        self.adenine_indices_to_set_1 = torch.tensor([1, 5, 21])
        self.guanine_indices_to_set_1 = torch.tensor([1, 5, 22])
        self.nucleotide_mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3} 
        
        # freeze RF2NA
        for param in self.folding_model.model.parameters():
            param.requires_grad=False
            param.data = param.data.float()

        self.interpolant = Interpolant()

        self.epoch_rmsds = []
        self.epoch_rna_aars = []

        self.val_epoch_rmsds = []
        self.val_epoch_rna_aars = []
            
    def model_step(self, pdb_id, data, fold_cplx=False):
        
        # noise RNA coords (gaussian prior)
        true_crds = torch.cat((data["prot_coords"][0], data["rna_coords"][0]),axis=0)[None,:]
        noise_mask = torch.cat((torch.zeros((1, data["prot_coords"].shape[1])), torch.ones((1, data["rna_coords"].shape[1]))), dim=1)
        noisy_rna_crds, timestep = self.interpolant.corrupt_coords(true_crds, noise_mask)
        data["rna_coords"] = noisy_rna_crds

        # run through IF (with timestep)
        seq_loss, rna_recovery_rate, pred_rna_seq, pred_one_hot = self.denoise_model.model_step(data, timestep, noisy=True)
        
        # IF logits are in order A, G, C, U (swapping G and C for RF2NA)
        pred_one_hot = pred_one_hot.cpu()
        pred_one_hot = torch.cat((torch.zeros(pred_one_hot.shape[0],27), 
                                 pred_one_hot[:,0:1], pred_one_hot[:,2:3],
                                 pred_one_hot[:,1:2], pred_one_hot[:,3:], 
                                 torch.zeros(pred_one_hot.shape[0],1)), dim=-1)[None,:]

        # run through RF2NA
        pred_rna_bb_crds_aligned, struct_loss, rna_rmsd, plddt = self.run_folding(pdb_id, data, pred_rna_seq, true_crds, pred_one_hot, fold_cplx=fold_cplx)

        # supervise on cross entropy and MSE
        return seq_loss, struct_loss, rna_recovery_rate, pred_rna_seq, rna_rmsd
            
    def training_step(self, batch):
        pdb_id, data = batch
        pdb_id = pdb_id[0]

        # zero center RNA coords (leaking this in train)
        rna_centroid = torch.mean(data["rna_coords"], dim=(1,2))
        data["rna_coords"] -= rna_centroid
        data["prot_coords"] -= rna_centroid

        seq_loss, struct_loss, rna_recovery_rate, pred_rna_seq, rna_rmsd = self.model_step(pdb_id, data, fold_cplx=False)
        outputs = {"rna_recovery_rate": rna_recovery_rate, "rna_rmsd": rna_rmsd}
        self.epoch_rmsds.append(outputs["rna_rmsd"])
        self.epoch_rna_aars.append(outputs["rna_recovery_rate"])
        if seq_loss is None or struct_loss is None:
            return torch.Tensor([0]).to("cuda")
        else:
            return seq_loss + struct_loss

    def validation_step(self, batch):
        pdb_id, data = batch
        pdb_id = pdb_id[0]

        # zero center RNA coords (leaking this in train on purpose)
        rna_centroid = torch.mean(data["rna_coords"], dim=(1,2))
        data["rna_coords"] -= rna_centroid
        data["prot_coords"] -= rna_centroid

        seq_loss, struct_loss, rna_recovery_rate, pred_rna_seq, rna_rmsd = self.model_step(pdb_id, data, fold_cplx=False)
        outputs = {"rna_recovery_rate": rna_recovery_rate, "rna_rmsd": rna_rmsd}
        self.val_epoch_rmsds.append(outputs["rna_rmsd"])
        self.val_epoch_rna_aars.append(outputs["rna_recovery_rate"])
        if seq_loss is None or struct_loss is None:
            self.log("val_loss", torch.Tensor([0]).to("cuda"))
        else:
            self.log("val_loss", seq_loss + struct_loss)

    def predict_step(self, batch):
        pdb_id, data = batch
        pdb_id = pdb_id[0]

        if not os.path.exists(f"/home/dnori/rna-design/src/scripts/output_pdbs_noprot_seq_sim/{pdb_id}"):
            os.mkdir(f"/home/dnori/rna-design/src/scripts/output_pdbs_noprot_seq_sim/{pdb_id}")

        for i in range(1):

            print(pdb_id)

            if not os.path.exists(f"/home/dnori/rna-design/src/scripts/output_pdbs_noprot_seq_sim/{pdb_id}/sample_{i}"):
                os.mkdir(f"/home/dnori/rna-design/src/scripts/output_pdbs_noprot_seq_sim/{pdb_id}/sample_{i}")
                os.mkdir(f"/home/dnori/rna-design/src/scripts/output_pdbs_noprot_seq_sim/{pdb_id}/sample_{i}/traj")
        
            true_cplx_crds = torch.cat((data["prot_coords"][0], data["rna_coords"][0]),axis=0)[None,:]
            noise_mask = torch.cat((torch.zeros((1, data["prot_coords"].shape[1])), torch.ones((1, data["rna_coords"].shape[1]))), dim=1)
            
            # initial dock guess
            rand_rna_seq = "A" * len(data["rna_seq"][0])
            rand_one_hot = torch.zeros((1,len(rand_rna_seq),32))
            rand_one_hot[:,:,27] = 1
            with torch.inference_mode(False):
                pred_docked_cplx, struct_loss, rna_rmsd, plddt = self.run_folding(pdb_id, data, rand_rna_seq, true_cplx_crds, rand_one_hot, fold_cplx=True) # passing true for RMSD calc
                if pred_docked_cplx is None:
                    outputs = {"rna_rmsd": 0, "rna_recovery": 0, "pdb_ids": pdb_id, "pred_seqs": "X"}
                    self.log_to_csv(outputs)
                    return None

            # align coords to dock guess
            true_rna_aligned, _, _ = frame_utils.kabsch(data["rna_coords"].view(1,-1,3), pred_docked_cplx[noise_mask.bool()[0]].view(1,-1,3)) # [B, 3*L, 3]
            true_prot_aligned, _, _ = frame_utils.kabsch(data["prot_coords"].view(1,-1,3), pred_docked_cplx[~noise_mask.bool()[0]].view(1,-1,3))
            data["rna_coords"] = true_rna_aligned.view(1,-1,3,3)
            data["prot_coords"] = true_prot_aligned.view(1,-1,3,3)

            # zero center RNA coords (with predicted RNA centroid)
            rna_centroid = torch.mean(data["rna_coords"], dim=(1,2))
            data["rna_coords"] -= rna_centroid
            data["prot_coords"] -= rna_centroid
            realigned_true_cplx_crds = torch.cat((data["prot_coords"][0], data["rna_coords"][0]),axis=0)[None,:]

            # sample prior
            trans_0 = _centered_gaussian(1, int(noise_mask.sum().item()*3), true_cplx_crds.device) * NM_TO_ANG_SCALE
            prior_crds = trans_0.view(1, -1, 3, 3)
            prior_cplx_crds = torch.cat((realigned_true_cplx_crds[:,~noise_mask[0].bool()], prior_crds), dim=1) # RNA noise centered at zero

            ts = torch.linspace(0.01, 1.0, 5)
            t_1 = ts[0]

            prot_traj = [prior_cplx_crds]
            idx = 0
            rmsd_list = []
            for t_2 in ts[1:]:

                t_1_tensor = torch.Tensor([t_1]).to(true_cplx_crds.device)

                crds_t_1 = prot_traj[-1]
                trans_t_1 = crds_t_1[:,noise_mask.bool()[0]].view(1,-1,3)
                data["prot_coords"] = crds_t_1[:,~noise_mask.bool()[0]]
                data["rna_coords"] = crds_t_1[:,noise_mask.bool()[0]]

                # run through IF (with timestep)
                seq_loss, rna_recovery_rate, pred_rna_seq, pred_one_hot, eval_perplexity, rank_perplexity = self.denoise_model.predict_step(batch, data=data, timestep=t_1_tensor, in_rnaflow=True)
      
                # IF logits are in order A, G, C, U (swapping G and C for RF2NA)
                pred_one_hot = pred_one_hot.cpu()
                pred_one_hot = torch.cat((torch.zeros(pred_one_hot.shape[0],27), 
                                        pred_one_hot[:,0:1], pred_one_hot[:,2:3],
                                        pred_one_hot[:,1:2], pred_one_hot[:,3:], 
                                        torch.zeros(pred_one_hot.shape[0],1)), dim=-1)[None,:]
                

                # run through RF2NA
                with torch.inference_mode(False):
                    pred_bb_crds, struct_loss, rna_rmsd, plddt = self.run_folding(pdb_id, data, pred_rna_seq, true_cplx_crds, pred_one_hot, fold_cplx=True) # passing true for RMSD calc

                # zero center pred RNA
                rna_centroid = torch.mean(pred_bb_crds[noise_mask[0].bool()], dim=(0,1))
                pred_bb_crds_zeroed = pred_bb_crds - rna_centroid
                rmsd_list.append(rna_rmsd)
                pdb_utils.save_rna_pdb(pred_bb_crds_zeroed[noise_mask[0].bool()], data["rna_seq"][0], f"/home/dnori/rna-design/src/scripts/output_pdbs_noprot_seq_sim/{pdb_id}/sample_{i}/traj/t_{idx}.pdb")

                # interpolate the RNA coords (all zero centered)
                pred_trans_1 = pred_bb_crds_zeroed[noise_mask[0].bool()].contiguous().view(1,-1,3)
                d_t = t_2 - t_1
                trans_t_2 = self.interpolant._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)

                # align true prot to pred prot
                true_prot_aligned, _, _ = frame_utils.kabsch(true_cplx_crds[:,~noise_mask[0].bool()].view(1,-1,3), pred_bb_crds_zeroed[~noise_mask.bool()[0]].view(1,-1,3))
                true_prot_aligned = true_prot_aligned.view(1,-1,3,3)
                crds_t_2 = torch.cat((true_prot_aligned, trans_t_2.view(1, -1, 3, 3)), dim=1)

                prot_traj.append(crds_t_2)
                t_1 = t_2
                idx+=1

            # Final step
            t_1 = ts[-1]
            t_1_tensor = torch.Tensor([t_1]).to(self.device)
            crds_t_1 = prot_traj[-1]
            trans_t_1 = crds_t_1[:,noise_mask.bool()[0]].view(1,-1,3)
            data["rna_coords"] = crds_t_1[:,noise_mask.bool()[0]]

            # run through IF (with timestep)
            seq_loss, final_rna_recovery_rate, final_pred_rna_seq, pred_one_hot, eval_perplexity, rank_perplexity = self.denoise_model.predict_step(batch, data=data, timestep=t_1_tensor, in_rnaflow=True)

            # IF logits are in order A, G, C, U (swapping G and C for RF2NA)
            pred_one_hot = pred_one_hot.cpu()
            pred_one_hot = torch.cat((torch.zeros(pred_one_hot.shape[0],27), 
                                    pred_one_hot[:,0:1], pred_one_hot[:,2:3],
                                    pred_one_hot[:,1:2], pred_one_hot[:,3:], 
                                    torch.zeros(pred_one_hot.shape[0],1)), dim=-1)[None,:]

            # run through RF2NA
            with torch.inference_mode(False):
                pred_bb_crds, final_struct_loss, final_rna_rmsd, final_plddt = self.run_folding(pdb_id, data, pred_rna_seq, true_cplx_crds, pred_one_hot, fold_cplx=True)
            
            # zero center pred complex on pred RNA
            rna_centroid = torch.mean(pred_bb_crds[noise_mask[0].bool()], dim=(0,1))
            pred_bb_crds_zeroed = pred_bb_crds - rna_centroid
            rmsd_list.append(final_rna_rmsd)
            pdb_utils.save_rna_pdb(pred_bb_crds_zeroed[noise_mask[0].bool()], data["rna_seq"][0], f"/home/dnori/rna-design/src/scripts/output_pdbs_noprot_seq_sim/{pdb_id}/sample_{i}/traj/t_{idx}.pdb")

            # align true prot to pred prot
            true_prot_aligned, _, _ = frame_utils.kabsch(true_cplx_crds[:,~noise_mask[0].bool()].view(1,-1,3), pred_bb_crds_zeroed[None,~noise_mask.bool()[0]].view(1,-1,3))
            true_prot_aligned = true_prot_aligned.view(1,-1,3,3)

            # final complex crds
            final_pred_cplx_crds = torch.cat((true_prot_aligned, pred_bb_crds[None,noise_mask[0].bool()]), dim=1)
            pdb_utils.save_cplx_pdb(final_pred_cplx_crds[0], data["prot_seq"][0], data["rna_seq"][0], f"/home/dnori/rna-design/src/scripts/output_pdbs_noprot_seq_sim/{pdb_id}/sample_{i}/final_cplx.pdb")
            
            print(final_rna_rmsd, final_rna_recovery_rate.item())
            
            outputs = {"rna_rmsd": final_rna_rmsd, "rna_recovery": final_rna_recovery_rate.item(), "pdb_ids": pdb_id, "pred_seqs": final_pred_rna_seq, "rmsd_list": rmsd_list, "plddt": final_plddt, "eval_perplexity": eval_perplexity, "rank_perplexity": rank_perplexity}
            self.log_to_csv(outputs)
    
    def on_train_epoch_end(self):
        avg_rna_rmsd = sum(self.epoch_rmsds)/len(self.epoch_rmsds)
        avg_rna_aar = sum(self.epoch_rna_aars)/len(self.epoch_rna_aars)

        logs = {"epoch": self.current_epoch, "avg_rna_rmsd": avg_rna_rmsd, "avg_rna_aar": avg_rna_aar}
        self.log_to_csv(logs)

        self.epoch_rmsds = []
        self.epoch_rna_aars = []

    def on_val_epoch_end(self):
        avg_rna_rmsd = sum(self.val_epoch_rmsds)/len(self.val_epoch_rmsds)
        avg_rna_aar = sum(self.val_epoch_rna_aars)/len(self.val_epoch_rna_aars)

        logs = {"epoch": self.current_epoch, "avg_rna_rmsd": avg_rna_rmsd, "avg_rna_aar": avg_rna_aar}
        self.log_to_csv(logs)

        self.val_epoch_rmsds = []
        self.val_epoch_rna_aars = []

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.denoise_model.model.parameters(),
            lr=0.001
        )
    
    def featurize(self, pdb_id: str, rna_seq_input: str = None, pred_one_hot: torch.Tensor = None, fold_cplx: bool = True):
        prot_a3m = os.path.join(RF_DATA_FOLDER, pdb_id, "prot.a3m")
        prot_hhr = os.path.join(RF_DATA_FOLDER, pdb_id, "prot.hhr")
        prot_atab = os.path.join(RF_DATA_FOLDER, pdb_id, "prot.atab")
        rna_afa = os.path.join(RF_DATA_FOLDER, pdb_id, "rna.afa")
        if fold_cplx:
            inputs = f"P:{prot_a3m}:{prot_hhr}:{prot_atab} R:{rna_afa}"
        else:
            inputs = f"R:{rna_afa}"
        outs = self.folding_model.prep_inputs(inputs, ffdb=None, rna_seq_input=rna_seq_input, pred_one_hot=pred_one_hot)
        return outs
    
    def run_folding(self, pdb_id, data, rna_seq, true_crds=None, pred_one_hot=None, fold_cplx=True, grk2=False):
        if true_crds is None:
            true_crds = torch.cat((data["prot_coords"][0], data["rna_coords"][0]),axis=0)[None,:].to("cuda")

        rna_mask = torch.cat((torch.zeros((1, data["prot_coords"].shape[1])), torch.ones((1, data["rna_coords"].shape[1]))), dim=1)

        rna_codes = torch.tensor([self.nucleotide_mapping[n] for n in rna_seq])
        backbone_mask_tensor = torch.zeros(len(rna_seq), 36)

        backbone_mask_tensor[torch.nonzero(rna_codes==1), self.pyrimidine_indices_to_set_1] = 1
        backbone_mask_tensor[torch.nonzero(rna_codes==3), self.pyrimidine_indices_to_set_1] = 1
        backbone_mask_tensor[torch.nonzero(rna_codes==0), self.adenine_indices_to_set_1] = 1
        backbone_mask_tensor[torch.nonzero(rna_codes==2), self.guanine_indices_to_set_1] = 1
        if fold_cplx:
            backbone_mask_tensor = torch.cat((torch.zeros((len(data["prot_seq"][0]), 36)), backbone_mask_tensor), dim=0)
            backbone_mask_tensor[:len(data["prot_seq"][0]),:3] = 1
        backbone_mask_tensor = backbone_mask_tensor[:,:,None].repeat((1,1,3)).to("cuda")

        outs = self.featurize(pdb_id, rna_seq, pred_one_hot, fold_cplx=fold_cplx)

        if len(outs) == 4:
            return outs
        else:
            seq_i, pred_crds, logit_pae, mask_t_2d, same_chain, plddt = self.folding_model.run_rf_module(*outs)

            if not fold_cplx: # train
                
                pred_bb_crds = torch.masked_select(pred_crds[0], backbone_mask_tensor.bool()).reshape((backbone_mask_tensor.shape[0],3,3))

                if grk2:
                    rna_mask[:,50] = 0
                    rna_mask[:,65:74] = 0
                    pred_bb_crds = pred_bb_crds[rna_mask.bool()[0,50:]]

                pred_bb_crds, _, _ = frame_utils.kabsch(pred_bb_crds[None,:].contiguous().view(1,-1,3), true_crds[:,rna_mask.bool()[0]].view(1,-1,3))
                pred_bb_crds = pred_bb_crds.view(1,-1,3,3)

                rna_aligned_rna_mse = self.mse_loss(pred_bb_crds, true_crds[0,rna_mask.bool()[0]])
                rna_aligned_rna_rmsd = torch.sqrt(rna_aligned_rna_mse).cpu().item()
                plddt = torch.mean(plddt).cpu().item()
            
            else: # inference
                try:
                    pred_bb_crds = torch.masked_select(pred_crds[0], backbone_mask_tensor.bool()).reshape((backbone_mask_tensor.shape[0],3,3))

                    pred_bb_crds_aligned, _, _ = frame_utils.kabsch(pred_bb_crds[None].view(1,-1,3), true_crds[:].view(1,-1,3)) # just for MSE
                    pred_bb_crds_aligned = pred_bb_crds_aligned.view(1,-1,3,3)
                    cplx_aligned_cplx_mse = self.mse_loss(pred_bb_crds_aligned, true_crds[0])
                    cplx_aligned_cplx_rmsd = torch.sqrt(cplx_aligned_cplx_mse).cpu().item()
                    plddt = torch.mean(plddt).cpu().item()
                    return pred_bb_crds, cplx_aligned_cplx_mse, cplx_aligned_cplx_rmsd, plddt

                except Exception as e:
                    print(e)
                    print("template mismatch")
                    return None, None, None, None

            return pred_bb_crds, rna_aligned_rna_mse, rna_aligned_rna_rmsd, plddt
        
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