from Bio.PDB import PDBIO, Model, Chain, Atom, Residue, Structure

def save_cplx_pdb(coordinates, prot_sequence, rna_sequence, filename):
    prot_sequence = [char for char in prot_sequence]
    rna_sequence = [char for char in rna_sequence]
    if coordinates.shape[1] != 3 and coordinates.shape[2] != 3:
        raise ValueError("Input tensor shape should be (N, 3, 3)")

    structure = Structure.Structure('example')
    model = Model.Model(0)
    structure.add(model)

    amino_acid_chain_id = "A"
    nucleotide_chain_id = "B"

    amino_acid_chain = Chain.Chain(amino_acid_chain_id)
    nucleotide_chain = Chain.Chain(nucleotide_chain_id)

    for i in range(len(prot_sequence)):
        
        aa = prot_sequence[i]
        aa_coord_ca = coordinates[i, 0, :]
        aa_coord_n = coordinates[i, 1, :]
        aa_coord_c = coordinates[i, 2, :]
        
        amino_acid = Residue.Residue((' ', i + 1, ' '), aa, ' ')
        amino_acid.add(Atom.Atom('CA', aa_coord_ca, 1.0, 0.0, ' ', 'CA', i + 1, element="C"))
        amino_acid.add(Atom.Atom('N', aa_coord_n, 1.0, 0.0, ' ', 'N', i + 1))
        amino_acid.add(Atom.Atom('C', aa_coord_c, 1.0, 0.0, ' ', 'C', i + 1))
        amino_acid_chain.add(amino_acid)

    model.add(amino_acid_chain)

    for j in range(len(rna_sequence)):
        
        idx = j + len(prot_sequence)
        nt = rna_sequence[j]
        nt_coord_P = coordinates[idx, 0, :]
        nt_coord_C = coordinates[idx, 1, :]
        nt_coord_N = coordinates[idx, 2, :]
        
        nucleotide = Residue.Residue((' ', idx + 1, ' '), nt, ' ')
        nucleotide.add(Atom.Atom('P', nt_coord_P, 1.0, 0.0, ' ', 'P', idx + 1))
        nucleotide.add(Atom.Atom("C4'", nt_coord_C, 1.0, 0.0, ' ', "C4'", idx + 1, element="C"))
        if nt in ["C", "U"]:
            nucleotide.add(Atom.Atom('N1', nt_coord_N, 1.0, 0.0, ' ', 'N1', idx + 1, element="N"))
        else:
            nucleotide.add(Atom.Atom('N9', nt_coord_N, 1.0, 0.0, ' ', 'N9', idx + 1, element="N"))
        nucleotide_chain.add(nucleotide)

    model.add(nucleotide_chain)

    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)

# whole backbone
def save_rna_pdb(coordinates, rna_sequence, filename):
    rna_sequence = [char for char in rna_sequence]
    if coordinates.shape[1] != 3 and coordinates.shape[2] != 3:
        raise ValueError("Input tensor shape should be (N, N, 3)")

    structure = Structure.Structure('example')
    model = Model.Model(0)
    structure.add(model)

    nucleotide_chain_id = "A"

    nucleotide_chain = Chain.Chain(nucleotide_chain_id)


    for j in range(len(rna_sequence)):

        nt = rna_sequence[j]
        nt_coord_P = coordinates[j, 0, :]
        nt_coord_C = coordinates[j, 1, :]
        nt_coord_N = coordinates[j, 2, :]
        
        nucleotide = Residue.Residue((' ', j + 1, ' '), nt, ' ')
        nucleotide.add(Atom.Atom('P', nt_coord_P, 1.0, 0.0, ' ', 'P', j + 1))
        nucleotide.add(Atom.Atom("C4'", nt_coord_C, 1.0, 0.0, ' ', "C4'", j + 1, element="C"))
        if nt in ["C", "U"]:
            nucleotide.add(Atom.Atom('N1', nt_coord_N, 1.0, 0.0, ' ', 'N1', j + 1, element="N"))
        else:
            nucleotide.add(Atom.Atom('N9', nt_coord_N, 1.0, 0.0, ' ', 'N9', j + 1, element="N"))
        nucleotide_chain.add(nucleotide)

    model.add(nucleotide_chain)

    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)