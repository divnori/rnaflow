"""
Processes select.csv from https://beta.nakb.org/download.html
Filters for protein-RNA complexes, saves appropriate PDB complexes
Pickles graph representations of protein + RNA complexes,
generates binary mask over structure based on distance to RNA
"""
import argparse
from Bio.PDB.PDBParser import PDBParser
from Bio import SeqIO
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import os
import random
from sklearn.cluster import DBSCAN
from scipy.signal import convolve
from sklearn.model_selection import train_test_split
import subprocess
from tqdm import tqdm
import urllib.request

import sys
sys.path.append("/home/dnori/rna-design/geometric-rna-design-v0/")

protein_letters_3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def load_and_filter(csv_file, pdb_out_dir):

    print(f"Processing {csv_file}.")

    # filter to protein-RNA complexes
    orig_df = pd.read_csv(csv_file)
    rna_df = orig_df[orig_df["polyclass"] == "Protein/RNA"][["pdbid"]].reset_index(drop=True)

    print(f"Number of Data Points: {len(rna_df)}.")

    # retrieve PDB complexes
    pdb_ids = list(set(rna_df["pdbid"].tolist()))
    i = 0
    for pdb_id in tqdm(pdb_ids):
        try:
            urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb_id}.pdb', 
            f'{pdb_out_dir}/{pdb_id}.pdb')
            urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb_id}.pdb', 
            f'{pdb_out_dir}/{pdb_id}.pdb')
            i+=1
        except:
            print(f"PDB not found for {pdb_id}.")

    print(f"Saved {i} complexes in {pdb_out_dir}.")

def calculate_distance_matrix(arr1, arr2):
    """
    Calculate the pairwise Euclidean distance matrix between two arrays.
    """
    return np.linalg.norm(arr1[:, np.newaxis] - arr2, axis=2)

def get_rna_seq_and_coords(structure):
    # Return sequence and coarse-grained coordinates array given PDB structure    
    seq = ""
    coords = []
    model = structure[0]
    chain_ids = []
    for chain in model:
        for residue in chain:
            print(residue.resname)
            if residue.resname in ["A", "G", "C", "U"]:
                if "P" not in residue: continue
                P_coord = residue["P"].coord

                if "C4'" not in residue: continue
                C_coord = residue["C4'"].coord

                # Pyrimidine (C, U): P, C4', and N1
                if residue.resname in ["C", "U"]:
                    if "N1" not in residue: continue
                    N_coord = residue["N1"].coord
                # Purine (A, G): P, C4', and N9
                else:
                    if "N9" not in residue: continue
                    N_coord = residue["N9"].coord

                seq += residue.resname
                coords.append([P_coord, C_coord, N_coord])
                chain_ids.append(chain.id)
    
    assert len(seq) == len(coords)
    return seq, np.array(coords), chain_ids

def get_protein_seq_and_coords(structure):
    # Return sequence and coarse-grained coordinates array given PDB structure
    seq = ""
    coords = []
    model = structure[0]
    chain_ids = []
    for chain in model:
        for residue in chain:
            if residue.resname in protein_letters_3to1.keys():
                if "CA" not in residue: continue
                CA_coord = residue["CA"].coord

                if "N" not in residue: continue
                N_coord = residue["N"].coord

                if "C" not in residue: continue
                C_coord = residue["C"].coord

                seq += protein_letters_3to1[residue.resname]
                coords.append([N_coord, CA_coord, C_coord])
                chain_ids.append(chain.id)
    
    assert len(seq) == len(coords)
    return seq, np.array(coords), chain_ids

def find_contact_chains(filepath):
    p = PDBParser(PERMISSIVE=1, QUIET=True)
    structure = p.get_structure(filepath[-8:-4], os.path.join('pdbs',filepath))
    header = structure.header
    deposit_date = header.get('deposition_date')

    rna_seq, rna_full_coords, rna_chain_ids = get_rna_seq_and_coords(structure)
    protein_seq, protein_full_coords, protein_chain_ids = get_protein_seq_and_coords(structure)

    # rna_full_coords - [P, C4', N1/N9]
    # prot_full_coords - [N, CA, C]

    try:
        rna_c4_coords = rna_full_coords[:,1,:]
        protein_ca_coords = protein_full_coords[:,1,:]
    except:
        return None

    target_rna = calculate_distance_matrix(protein_ca_coords, rna_c4_coords)

    data = {}
    if np.sum(target_rna) > 0:
        contact_position = np.unravel_index(np.argmin(target_rna), target_rna.shape)
        protein_chain = protein_chain_ids[contact_position[0]].upper()
        rna_chain = rna_chain_ids[contact_position[1]].upper()

        if f"{filepath[-8:-4]}_{protein_chain}_{rna_chain}" not in data:
    
            prot_indices = np.array([1 if c==protein_chain else 0 for c in protein_chain_ids], dtype=bool)
            rna_indices = np.array([1 if c==rna_chain else 0 for c in rna_chain_ids], dtype=bool)

            if prot_indices.shape[0] > 50: # 50 residue crop
                prot_indices = np.zeros_like(prot_indices)
                prot_indices[contact_position[0]-25:contact_position[0]+25] = 1
                prot_indices = prot_indices.astype(bool)
            
            prot_chain_coords = protein_full_coords[prot_indices]
            rna_chain_coords = rna_full_coords[rna_indices]
            prot_chain_seq = ''.join([char for char, mask_value in zip(protein_seq, prot_indices.tolist()) if mask_value])
            rna_chain_seq = ''.join([char for char, mask_value in zip(rna_seq, rna_indices.tolist()) if mask_value])

            assert prot_chain_coords.shape[0] > 0 and prot_chain_coords.shape[0] == len(prot_chain_seq) and rna_chain_coords.shape[0] == len(rna_chain_seq) and "T" not in rna_seq and len(rna_chain_seq) >= 6 and len(rna_chain_seq) <= 128

            data[f"{filepath[-8:-4].upper()}_{protein_chain}_{rna_chain}"] = {"prot_coords": prot_chain_coords, "rna_coords": rna_chain_coords, "prot_seq": prot_chain_seq, "rna_seq": rna_chain_seq, "deposit_date": deposit_date}

    return data

def write_rf_files(dataset, fa=False, prot_msa=False, rna_msa=False):
    """
    .fa
    .a3m
    .hhr
    .atab
    .afa
    """
    with open(dataset, 'rb') as handle:
        dataset = pickle.load(handle)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    for dataset in [train_dataset, test_dataset]:
        for pdb_id, v in dataset.items():

            prot_chain_seq = v["prot_seq"]
            rna_chain_seq = v["rna_seq"]

            if not os.path.exists(f"rf_data/{pdb_id}"):
                os.makedirs(f"rf_data/{pdb_id}")

            if fa:
                with open(f"rf_data/{pdb_id}/prot.fa", 'w') as fasta_file:
                    fasta_file.write(f'>prot\n{prot_chain_seq}\n')
                with open(f"rf_data/{pdb_id}/rna.fa", 'w') as fasta_file:
                    fasta_file.write(f'>rna\n{rna_chain_seq}\n')
                    
            if prot_msa: # blank msa (just gt seq)
                with open(f"rf_data/{pdb_id}/prot.a3m", 'w') as msa_file:
                    msa_file.write(f'>prot\n{prot_chain_seq}\n')

            if rna_msa:
                with open(f"rf_data/{pdb_id}/rna.afa", 'w') as msa_file:
                    msa_file.write(f'>rna\n{rna_chain_seq}\n')

def save_splits(rf2na_test_csv, pdb_out_dir, split="rf2na"):
    """
    Saves train and test splits for RNAFlow
    """
    dataset = {"train": {}, "val": {}, "test": {}}
    prot_seqs = {}
    rna_seqs = {}
    pdb_labels = []
    data_list = []

    pdb_ids = os.listdir("data/rf_data")
    rf2na_cutoff = datetime.strptime("2023-04-13", "%Y-%m-%d")

    files = os.listdir(pdb_out_dir)
    og_test_pdbs = pd.read_csv(rf2na_test_csv)["tag"].tolist()
    og_test_pdbs = [x[:4].upper() for x in og_test_pdbs]

    for pdb_file in tqdm(files):
        try:
            data = find_contact_chains(pdb_file)
            if data is None or len(data) == 0:
                continue
        except Exception as e:
            continue

        if split == "rf2na":
            for k,v in data.items():
                deposit_date = datetime.strptime(v["deposit_date"], "%Y-%m-%d")
                if k.upper() not in pdb_ids:
                    print(f"Dropping {k.upper()}")
                    continue
                if k[:4].upper() in og_test_pdbs or deposit_date > rf2na_cutoff:
                    dataset["test"][k.upper()] = v
                else:
                    pdb_labels.append(k.upper())
                    data_list.append(v)

        elif split == "seq_sim":
            for k,v in data.items():
                if k.upper() not in pdb_ids:
                    print(f"Dropping {k.upper()}")
                    continue
                prot_seq = v["prot_seq"]
                prot_seqs[k.upper()] = prot_seq
                rna_seqs[k.upper()] = v["rna_seq"]
                pdb_labels.append(k.upper())
                data_list.append((k,v))

    if split == "seq_sim":

        with open("data/rna_seq_clusters.pkl", 'rb') as handle:
            clusters = pickle.load(handle)
        
        last_cluster_index = max(clusters.values())
        test_clusters = random.sample(range(last_cluster_index), int(0.2*last_cluster_index))
        val_clusters = test_clusters[:len(test_clusters)//2]
        test_clusters = test_clusters[len(test_clusters)//2:]

        for it in data_list:
            k,v = it
            if k in clusters:
                cluster_idx = clusters[k]
                if cluster_idx in test_clusters:
                    dataset["test"][k.upper()] = v
                elif cluster_idx in val_clusters:
                    dataset["val"][k.upper()] = v
                else:
                    dataset["train"][k.upper()] = v
            else:
                dataset["train"][k.upper()] = v

    elif split == "rf2na":
        train_ids, val_ids, train_data, val_data = train_test_split(pdb_labels, data_list, test_size=0.1, random_state=42)
        for i in range(len(train_ids)):
            dataset["train"][train_ids[i].upper()] = train_data[i]
        for j in range(len(val_ids)):
            dataset["val"][val_ids[j].upper()] = val_data[j]

    print(f"Number in train = {len(dataset['train'])}")
    print(f"Number in val = {len(dataset['val'])}")
    print(f"Number in test = {len(dataset['test'])}")

    with open(f'{split}_split_dataset.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDB files and generate dataset")
    parser.add_argument("--pdbbind_csv", type=str, default="rnaflow/data/pdbbind_na.csv", help="CSV file containing PDB IDs from PDBBind")
    parser.add_argument("--pdb_out_dir", type=str, default="rnaflow/data/pdbs", help="Directory to save PDB files")
    parser.add_argument("--rf2na_test_csv", type=str, default="rnaflow/data/rf2na_test_val_set.csv", help="CSV file containing list of PDBs held out from rf2na training")
    parser.add_argument("--dataset", type=str, default="rnaflow/data/rf2na_dataset.pickle", help="Path to save the dataset")
    args = parser.parse_args()

    load_and_filter(args.pdbbind_csv, args.pdb_out_dir)
    save_splits(args.rf2na_test_csv, args.pdb_out_dir, split="rf2na")
    write_rf_files(args.dataset, fa=True, prot_msa=False, rna_msa=True)
    subprocess.run(["python", "prot_msa.py"])
    subprocess.run(['bash', 'rnaflow/data/run_RF2NA_mod.sh'])