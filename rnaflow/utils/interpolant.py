"""
Many functions from https://github.com/microsoft/frame-flow/blob/main/data/interpolant.py
"""

import copy
import numpy as np
import os
import random
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
import sys
import torch

network_path = os.path.join('RoseTTAFold2NA/network')
sys.path.append(network_path)

import so3_utils
import frame_utils

NM_TO_ANG_SCALE = 10.0

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )

class Interpolant:

    def __init__(self):
        if (torch.cuda.is_available()):
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self._igso3 = None

    def sample_t(self):
        # return random.uniform(0,1)
        return 0.01
    
    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = frame_utils.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
    
    def _corrupt_trans(self, trans_1, t, res_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self.device)
        trans_0 = trans_nm_0 * NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self.device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self.device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        scaling = 1 / (1 - t)
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)
    
    def corrupt_coords(self, true_cplx_coords, noise_mask):
        t = torch.Tensor([self.sample_t()]).to(self.device)
        trans = true_cplx_coords[:,noise_mask.bool()[0]].view(1,-1,3)
        corrupt_trans = self._corrupt_trans(trans, t, 
                                            torch.ones((1, trans.shape[1])).to(self.device))
        corrupt_trans = corrupt_trans.view(1, -1, 3, 3)
        return corrupt_trans, t

    def sample(self, model, batch, loss_fn):
        pdb_id, data = batch
        pdb_id = pdb_id[0]
        true_cplx_crds = torch.cat((data["prot_coords"][0], data["rna_coords"][0]),axis=0)[None,:]
    
        noise_mask = torch.cat((torch.zeros((1, data["prot_coords"].shape[1])), torch.ones((1, data["rna_coords"].shape[1]))), dim=1).to(true_cplx_crds.device)
        true_cplx_crds -= torch.mean(true_cplx_crds[:,noise_mask.bool()[0]]) # subtract CoM of RNA from posterior

        # Prior
        trans_0 = _centered_gaussian(1, int(noise_mask.sum().item()), self.device) * NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(1, int(noise_mask.sum().item()), self.device)
        prior_crds = frame_utils.frames_to_coords(rotmats_0, trans_0)[:,:,:3]
        prior_cplx_crds = torch.cat((true_cplx_crds[:,~noise_mask[0].bool()], prior_crds), dim=1) # RNA noise centered at zero

        ts = torch.linspace(0.01, 1.0, 5)
        t_1 = ts[0]

        prot_traj = [prior_cplx_crds]
        clean_traj = []
        for t_2 in ts[1:]:

            t_1_tensor = torch.Tensor([t_1]).to(self.device)

            crds_t_1 = prot_traj[-1]
            rots_t_1, trans_t_1 = frame_utils.coords_to_frames(true_cplx_crds, noise_mask)

            with torch.inference_mode(False):
                seq_i, pred_crds, logit_pae, mask_t_2d, same_chain = model.forward(pdb_id, crds_t_1, t_1_tensor)

            pred_crds = pred_crds[:,:,:3].clone()
            clean_traj.append(pred_crds.detach().cpu())

            rna_rmsd = torch.sqrt(loss_fn(pred_crds[0,noise_mask.bool()[0]], true_cplx_crds[0,noise_mask.bool()[0]])).cpu().detach().numpy().item()
            cplx_rmsd = torch.sqrt(loss_fn(pred_crds, true_cplx_crds))

            d_t = t_2 - t_1
            pred_rots_1, pred_trans_1 = frame_utils.coords_to_frames(true_cplx_crds, noise_mask)
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1[:,noise_mask[0].bool()], trans_t_1[:,noise_mask[0].bool()])
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rots_1[:,noise_mask[0].bool()], rots_t_1[:,noise_mask[0].bool()])
            crds_t_2 = frame_utils.frames_to_coords(rotmats_t_2, trans_t_2)[:,:,:3]
            crds_t_2 = torch.cat((true_cplx_crds[:,~noise_mask[0].bool()], crds_t_2), dim=1) #flow matching only on RNA
        
        # Final step
        t_1 = ts[-1]
        t_1_tensor = torch.Tensor([t_1]).to(self.device)
        crds_t_1 = prot_traj[-1]
        rots_t_1, trans_t_1 = frame_utils.coords_to_frames(true_cplx_crds, noise_mask)

        with torch.inference_mode(False):
            seq_i, pred_crds, logit_pae, mask_t_2d, same_chain = model.forward(pdb_id, crds_t_1, t_1_tensor)

        final_pred_crds = pred_crds[:,:,:3].clone()
        clean_traj.append(final_pred_crds.detach().cpu())
        prot_traj.append(final_pred_crds)

        rna_rmsd = torch.sqrt(loss_fn(final_pred_crds[0,noise_mask.bool()[0]], true_cplx_crds[0,noise_mask.bool()[0]])).cpu().detach().numpy().item()
        cplx_rmsd = torch.sqrt(loss_fn(final_pred_crds, true_cplx_crds))

        true_cplx_crds  += torch.mean(true_cplx_crds[:,noise_mask.bool()[0]])
        final_pred_crds += torch.mean(true_cplx_crds[:,noise_mask.bool()[0]])

        return {"cplx_rmsd": cplx_rmsd, "rna_rmsd": rna_rmsd, "pdb_id": pdb_id}, final_pred_crds, true_cplx_crds