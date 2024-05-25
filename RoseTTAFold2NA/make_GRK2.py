import sys
import random
import os

pos = "GGCAGACCAUACGGGAGAGAAACUUGCC"
sspair = {
    27:0, 26:1, 25:2, 24:3, 6:13, 7:12, 
}
comp = {
    "A": "U", "C": "G", "U": "A", "G": "C"
}

for seed in range(8):
    random.seed(seed)
    cands = []
    for _ in range(1250):
        seq = [random.choice("ACGU") for _ in pos]
        for k,v in sspair.items():
            seq[k] = comp[seq[v]]
        seq = ''.join(seq)
        cands.append(seq)

    for i,seq in enumerate(cands):
        os.makedirs(f"aptamer_data/GRK2_{seed}/{i+1}", exist_ok=True)
        with open(f"aptamer_data/GRK2_{seed}/{i+1}/RNA.fa", 'w') as f:
            print(f">{i+1}", file=f)
            print(seq, file=f)
