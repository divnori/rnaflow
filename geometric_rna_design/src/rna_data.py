import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

from data_utils import *
from featurisation import *

class RNADesignDataset(data.Dataset):
    '''Multi-state RNA Design Dataset

    Builds 3-bead coarse grained representation of RNA backbone: (P, C4', N1 or N9).

    Returned graphs are of type `torch_geometric.data.Data` with attributes:
    - pos        C4' carbon coordinates, shape [n_nodes, n_conf, 3]
    - seq        sequence converted to int tensor, shape [n_nodes]
    - node_s     node scalar features, shape [n_nodes, n_conf, 64] 
    - node_v     node vector features, shape [n_nodes, n_conf, 4, 3]
    - edge_s     edge scalar features, shape [n_edges, n_conf, 32]
    - edge_v     edge vector features, shape [n_edges, n_conf, 1, 3]
    - edge_index edge indices, shape [2, n_edges]
    - mask       node mask, `False` for nodes with missing data

    Arguments:
    :param data_list: List of data samples
    :param split: train/validation/test split
    :param radius: radial cutoff for drawing edges (currently not used)
    :param top_k: number of edges to draw per node as destination node
    :param num_rbf: number of radial basis functions
    :param num_posenc: number of positional encodings per edge
    :param num_conformers: maximum number of conformers sampled per sequence
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(
            self,
            data_list = [],
            split = 'train',
            radius = 4.5,
            top_k_internal = 10,
            top_k_external = 5,
            num_rbf = 16,
            num_posenc = 16,
            num_conformers = 1,
            device = 'cuda'
        ):
        super().__init__()

        self.data_list = data_list
        self.split = split
        self.radius = radius
        self.top_k_internal = top_k_internal
        self.top_k_external = top_k_external
        self.num_rbf = num_rbf
        self.num_posenc = num_posenc
        self.num_conformers = num_conformers
        self.device = device

        self.rna_letter_to_num = {'A': 0, 'G': 1, 'C': 2, 'U': 3}
        self.rna_num_to_letter = {v:k for k, v in self.rna_letter_to_num.items()}

        self.prot_letter_to_num = {'C': 4, 'D': 5, 'S': 6, 'Q': 7, 'K': 8, 'I': 9,
                       'P': 10, 'T': 11, 'F': 12, 'A': 13, 'G': 14, 'H': 15,
                       'E': 16, 'L': 17, 'R': 18, 'W': 19, 'V': 20, 
                       'N': 21, 'Y': 22, 'M': 23} 
        self.prot_num_to_letter = {v:k for k, v in self.prot_letter_to_num.items()}

        self.node_counts = [len(entry['seq']) * len(entry['coords_list']) for entry in self.data_list]
        print(f"{split}: {len(self.data_list)} samples")
    
    def __len__(self): 
        return len(self.data_list)
    
    def __getitem__(self, i): 
        return self._featurize(self.data_list[i])
    
    def _featurize(self, seq, coords_list, rna=True):
        """
        Use to construct individual protein and RNA graphs
        """
        with torch.no_grad():
            # Target sequence: num_res x 1
            if rna:
                seq = torch.as_tensor(
                    [self.rna_letter_to_num[residue] for residue in seq], 
                    device=self.device, 
                    dtype=torch.long
                )[:,None]

                token_representations = torch.zeros((seq.shape[0], 1, 21)).to(self.device) # masked
                token_representations[:,-1] = 1

            else:
                seq = torch.as_tensor(
                    [self.prot_letter_to_num[residue] for residue in seq], 
                    device=self.device, 
                    dtype=torch.long
                )[:,None]

                token_representations = F.one_hot(seq-4, num_classes=21).to(self.device) # first 20 rows are AA, 21st row is mask RNA

            # Mask extra coordinates if fewer than num_conf: num_res x num_conf
            mask_confs = np.ones((1,coords_list.shape[0]))
            mask_confs = torch.BoolTensor(mask_confs).repeat(len(seq), 1).to(self.device)

            # Mask missing values: num_res
            mask_coords = torch.isfinite(coords_list.sum(dim=(0,2,3)))
            coords_list[:, ~mask_coords] = np.inf
            
            # C4' coordinates as node positions: num_conf x num_res x 3
            coord_C_list = coords_list[:, :, 1]
            
            # Construct merged edge index
            edge_index = torch.LongTensor().to(self.device)
            for coord in coord_C_list:
                edge_index = torch.concat(
                    [edge_index, torch_cluster.knn_graph(coord, self.top_k_internal)],
                    dim = 1,
                )
            
            edge_index = torch_geometric.utils.coalesce(edge_index)

            # Node attributres: distances and displacement vectors along backbone
            fwd_dist, bck_dist, fwd_vec, bck_vec = get_backbone_dist_and_vec(coord_C_list)
            fwd_rbf = rbf(fwd_dist.permute(1, 0), D_count=self.num_rbf, device=self.device)  # num_res x num_conf x num_rbf
            bck_rbf = rbf(bck_dist.permute(1, 0), D_count=self.num_rbf, device=self.device)  # num_res x num_conf x num_rbf
            fwd_vec = fwd_vec.unsqueeze_(-2).permute(1, 0, 2, 3)         # num_res x num_conf x 1 x 3
            bck_vec = bck_vec.unsqueeze_(-2).permute(1, 0, 2, 3)         # num_res x num_conf x 1 x 3
            
            # Node attributres: distances and displacement within nucleotide
            CN_dist, CP_dist, CN_vec, CP_vec = get_C_to_NP_dist_and_vec(coords_list)
            CN_rbf = rbf(CN_dist.permute(1, 0), D_count=self.num_rbf, device=self.device)    # num_res x num_conf x num_rbf
            CP_rbf = rbf(CP_dist.permute(1, 0), D_count=self.num_rbf, device=self.device)    # num_res x num_conf x num_rbf
            CN_vec = CN_vec.unsqueeze_(-2).permute(1, 0, 2, 3)           # num_res x num_conf x 1 x 3
            CP_vec = CP_vec.unsqueeze_(-2).permute(1, 0, 2, 3)           # num_res x num_conf x 1 x 3
            
            # Reshape coord_C_list: num_res x num_conf x 3
            coord_C_list = coord_C_list.permute(1, 0, 2)
            
            # Edge displacement vectors: num_edges x num_conf x  3
            edge_vectors = coord_C_list[edge_index[0]] - coord_C_list[ edge_index[1]]
            # Edge RBF features: num_edges x num_conf x num_rbf
            edge_rbf = rbf(edge_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            # Edge positional encodings: num_edges x num_conf x num_posenc
            edge_posenc = get_posenc(edge_index, self.num_posenc, device=self.device).unsqueeze_(1).repeat(1, self.num_conformers, 1)
            
            node_s = torch.cat([fwd_rbf, bck_rbf, CN_rbf, CP_rbf, token_representations], dim=-1)
            node_v = torch.cat([fwd_vec, bck_vec, CN_vec, CP_vec], dim=-2)
            edge_s = torch.cat([edge_rbf, edge_posenc], dim=-1)
            edge_v = normalize(edge_vectors).unsqueeze(-2)
            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num,
                (node_s, node_v, edge_s, edge_v)
            )

            # NOTE: if we want to move to e3nn, we need to combine _s and _v features.
        
        # print(seq.shape, node_s.shape, node_v.shape, edge_s.shape, edge_v.shape, edge_index.shape, mask_coords.shape, mask_confs.shape)
        data = torch_geometric.data.Data(
            seq = seq,                  # num_res x 1
            node_s = node_s,            # num_res x num_conf x 85
            node_v = node_v,            # num_res x num_conf x 4 x 3
            edge_s = edge_s,            # num_edges x num_conf x 32
            edge_v = edge_v,            # num_edges x num_conf x 1 x 3
            edge_index = edge_index,    # 2 x num_edges
            mask_coords = mask_coords,  # num_res
            mask_confs = mask_confs,    # num_res x num_conf
        )
        return data, coords_list

    def _connect_graphs(self, protein_graph, rna_graph, prot_coords, rna_coords):
        
        # simply concat graphs to get node features and internal edge features
        seq = torch.cat((protein_graph.seq, rna_graph.seq))
        node_s = torch.cat((protein_graph.node_s, rna_graph.node_s))
        node_v = torch.cat((protein_graph.node_v, rna_graph.node_v))
        internal_edge_s = torch.cat((protein_graph.edge_s, rna_graph.edge_s))
        internal_edge_v = torch.cat((protein_graph.edge_v, rna_graph.edge_v))
        mask_coords = torch.cat((protein_graph.mask_coords, rna_graph.mask_coords))
        mask_confs = torch.cat((protein_graph.mask_confs, rna_graph.mask_confs))

        # add offset for internal edge index
        num_nodes1 = protein_graph.node_s.shape[0]
        num_nodes2 = rna_graph.node_s.shape[0]
        internal_edge_index = rna_graph.edge_index + torch.tensor([[num_nodes1], [num_nodes1]], dtype=torch.long, device=self.device)
        internal_edge_index = torch.cat((protein_graph.edge_index, internal_edge_index),dim=1)

        # CA coordinates as node positions: num_conf x num_res x 3
        prot_coord_C_list = prot_coords[:, :, 1]
        # C4' coordinates as node positions: num_conf x num_res x 3
        rna_coord_C_list = rna_coords[:, :, 1]

        # Construct external edge index
        # Each coord in prot graph connects to 5 RNA coords
        external_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        for i in range(prot_coord_C_list.shape[1]):
            distances = torch.cdist(prot_coord_C_list[:, i:i+1], rna_coord_C_list) # num_conf x num_res in protein x 1
            _, knn_indices = distances.topk(self.top_k_external, dim=2, largest=False) # num_conf x self.top_k_external x 1

            edge_indices = torch.stack([
                knn_indices.view(-1) + num_nodes1, # offset for protein
                torch.tensor([i] * knn_indices.shape[2], dtype=torch.long, device=self.device).view(-1),
            ])

            external_edge_index = torch.cat([external_edge_index, edge_indices], dim=1)

        # Reshape coord_C_list: num_res x num_conf x 3
        coord_C_list = torch.cat((prot_coord_C_list, rna_coord_C_list), dim=1)
        coord_C_list = coord_C_list.permute(1, 0, 2)
        
        # Edge displacement vectors: num_edges x num_conf x  3
        external_edge_vectors = coord_C_list[external_edge_index[0]] - coord_C_list[ external_edge_index[1]]
        # Edge RBF features: num_edges x num_conf x num_rbf
        external_edge_rbf = rbf(external_edge_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

        # Edge positional encodings: num_edges x num_conf x num_posenc [high value - not on same graph]
        external_edge_posenc = get_posenc(external_edge_index, self.num_posenc, device=self.device, same_graph=False, const=num_nodes1+num_nodes2).unsqueeze_(1).repeat(1, self.num_conformers, 1)

        external_edge_s = torch.cat([external_edge_rbf, external_edge_posenc], dim=-1)
        external_edge_v = normalize(external_edge_vectors).unsqueeze(-2)

        edge_index = torch.cat([internal_edge_index, external_edge_index], dim=1) # internal edges then external edges
        edge_s = torch.cat((internal_edge_s, external_edge_s))
        edge_v = torch.cat((internal_edge_v, external_edge_v))

        num_internal_edges = internal_edge_index.shape[1]
        mask_internal_edges = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=self.device)
        mask_internal_edges[:num_internal_edges] = True

        data = torch_geometric.data.Data(
            seq = seq,                  # num_res x 1
            node_s = node_s,            # num_res x num_conf x 85
            node_v = node_v,            # num_res x num_conf x 4 x 3
            edge_s = edge_s,            # num_edges x num_conf x 32
            edge_v = edge_v,            # num_edges x num_conf x 1 x 3
            edge_index = edge_index,    # 2 x num_edges
            mask_coords = mask_coords,  # num_res
            mask_confs = mask_confs,    # num_res x num_conf
            internal_edge_mask = mask_internal_edges
        )
        return data


class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_nodes=5000, shuffle=True):
        
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))  
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch
