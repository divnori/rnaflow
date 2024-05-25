import os
import sys
import torch
from torch_scatter import scatter_add, scatter

network_path = os.path.join('RoseTTAFold2NA/network')
sys.path.append(network_path)

from util import rigid_from_3_points
from util_module import XYZConverter
from chemical import NTOTALDOFS

converter = XYZConverter()

def coords_to_frames(coords, mask):
    N, CA, C = coords[:,:,0,:], coords[:,:,1,:], coords[:,:,2,:]
    Rs,Ts = rigid_from_3_points(N, CA, C, is_na=mask.bool())
    return Rs, Ts

def frames_to_coords(rots, trans):
    L = rots.shape[1]
    seq = torch.tensor([31 for _ in range(L)])[None,:] # fully masked
    alphas = torch.zeros((1,L,NTOTALDOFS,2))
    _, coords = converter.frames_to_na_coords(seq, rots, trans, alphas)
    return coords

@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """
    Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
    sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
    the reference can be given as a single structure and broadcasted. Returns the structure
    coordinates shifted to the geometric center and the batch structures rotated to match the
    reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
    atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions = center_zero(batch_positions, batch_indices)
    reference_positions = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter_add(
        batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0
    )

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated, reference_positions, rotation_matrices

def batch_align_structures(pos_1, pos_2, mask=None):
    if pos_1.shape != pos_2.shape:
        raise ValueError('pos_1 and pos_2 must have the same shape.')
    if pos_1.ndim != 3:
        raise ValueError(f'Expected inputs to have shape [B, N, 3]')
    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
        torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64) 
        * torch.arange(num_batch, device=device)[:, None]
    )
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        aligned_pos_1, aligned_pos_2, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2)
        aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
        aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
        return aligned_pos_1, aligned_pos_2, align_rots

    flat_mask = mask.reshape(-1).bool()
    _, _, align_rots = align_structures(
        flat_pos_1[flat_mask],
        flat_batch_indices[flat_mask],
        flat_pos_2[flat_mask]
    )
    aligned_pos_1 = torch.bmm(
        pos_1,
        align_rots
    )
    return aligned_pos_1, pos_2, align_rots

def kabsch(A, B):
    with torch.no_grad():
        a_mean = A.mean(dim=1, keepdims=True)
        b_mean = B.mean(dim=1, keepdims=True)
        A_c = A - a_mean
        B_c = B - b_mean
        # Covariance matrix
        H = torch.bmm(A_c.transpose(1,2), B_c)  # [B, 3, 3]
        U, S, V = torch.svd(H)
        # Flip
        sign = (torch.det(U) * torch.det(V) < 0.0)
        if sign.any():
            S[sign] = S[sign] * (-1)
            U[sign,:] = U[sign,:] * (-1)
        # Rotation matrix
        R = torch.bmm(V, U.transpose(1,2))  # [B, 3, 3]
        # Translation vector
        t = b_mean - torch.bmm(R, a_mean.transpose(1,2)).transpose(1,2)
        A_aligned = torch.bmm(R, A.transpose(1,2)).transpose(1,2) + t
        return A_aligned, R, t

def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> torch.Tensor:
    """
    Move the molecule center to zero for sparse position tensors.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

    means = scatter(pos, batch_indexes, dim=0, reduce="mean")
    return pos - means[batch_indexes]