from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import rdmolops
# import datamol as dm
import numpy as np
from rdkit.Chem import BRICS
from . import sascorer
from rdkit.Chem import QED
import torch

def get_qed(mol):
    return QED.qed(mol)


def get_sa(mol):
    return (10 - sascorer.calculateScore(mol)) / 9


def get_morgan_fingerprint(mol, radius=2, nbits=2048):
    """
    Get Morgan fingerprint of molecule.
    Returns None if parsing fails (mol=None).
    """
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return fp


def is_too_similar_to_children(new_smiles_fp, children_smiles_fps, threshold=0.8):
    """
    Check if new_smiles is too similar to any SMILES in children_smiles list
    based on Tanimoto similarity threshold.

    - new_smiles: New generated SMILES string
    - children_smiles: List of SMILES for all child nodes of current node
    - threshold: Similarity threshold, deemed "too similar" if exceeded

    Return:
      True  -> If too similar (or invalid SMILES), reject
      False -> If not similar or acceptable
    """
    new_fp = new_smiles_fp
    if new_fp is None:
        # If parsing fails, consider as unusable (or other handling as desired)
        return True

    for fp in children_smiles_fps:
        if fp is None:
            continue
        tanimoto = TanimotoSimilarity(new_fp, fp)
        if tanimoto >= threshold:
            return True
    return False


dummy = Chem.MolFromSmiles('[*]')


def mol_from_smiles(smi):
    smi = canonicalize(smi)
    return Chem.MolFromSmiles(smi)


def strip_dummy_atoms(mol):
    try:
        hydrogen = mol_from_smiles('[H]')
        mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
        mol = Chem.RemoveHs(mols[0])
    except Exception:
        return None
    return mol


def break_on_bond(mol, bond, min_length=3):
    if mol.GetNumAtoms() - bond <= min_length:
        return [mol]

    broken = Chem.FragmentOnBonds(
        mol, bondIndices=[bond],
        dummyLabels=[(0, 0)])

    res = Chem.GetMolFrags(
        broken, asMols=True, sanitizeFrags=False)

    return res


def get_size(frag):
    dummies = count_dummies(frag)
    total_atoms = frag.GetNumAtoms()
    real_atoms = total_atoms - dummies
    return real_atoms


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count


def mol_to_smiles(mol):
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)


def mols_to_smiles(mols):
    return [mol_to_smiles(m) for m in mols]


def canonicalize(smi, clear_stereo=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def fragment_recursive(mol, frags):
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))

        if bonds == []:
            frags.append(mol)
            return frags

        idxs, labs = list(zip(*bonds))

        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())

        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]

        # Only one bond is broken. If a fragment can be cut at two points, only one is cut, and the other remains integral with the shortened part
        broken = Chem.FragmentOnBonds(mol,
                                      bondIndices=[bond_idxs[0]],
                                      dummyLabels=[(0, 0)])
        head, tail = Chem.GetMolFrags(broken, asMols=True)
        # print(mol_to_smiles(head), mol_to_smiles(tail))
        frags.append(head)
        return fragment_recursive(tail, frags)
    except Exception:
        pass


def join_molecules(molA, molB):
    marked, neigh = None, None
    for atom in molA.GetAtoms():
        if atom.GetAtomicNum() == 0:
            marked = atom.GetIdx()
            if len(atom.GetNeighbors()) == 0:
                return None
            neigh = atom.GetNeighbors()[0]
            break
    neigh = 0 if neigh is None else neigh.GetIdx()

    if marked is not None:
        ed = Chem.EditableMol(molA)
        if neigh > marked:
            neigh = neigh - 1
        ed.RemoveAtom(marked)
        molA = ed.GetMol()

    joined = Chem.ReplaceSubstructs(
        molB, dummy, molA,
        replacementConnectionPoint=neigh,
        useChirality=False)[0]

    try:
        Chem.Kekulize(joined)
    except Exception:
        return None
    return joined


def reconstruct(frags, reverse=False):
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags

    mol = join_molecules(frags[0], frags[1])
    if mol is None:
        return None, frags
    for i, frag in enumerate(frags[2:]):
        # print(i, mol_to_smiles(frag), mol_to_smiles(mol))
        mol = join_molecules(mol, frag)
        if mol is None:
            break
        # print(i, mol_to_smiles(mol))
    if mol is None:
        return None, frags

    # Remove chirality information
    rdmolops.RemoveStereochemistry(mol)

    # see if there are kekulization/valence errors
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    smi = canonicalize(smi)
    if smi is None:
        return None, frags

    return mol, frags


def break_into_fragments(mol, smi):
    frags = []
    frags = fragment_recursive(mol, frags)

    if len(frags) == 0:
        return smi, np.nan, 0

    if len(frags) == 1:
        return smi, smi, 1

    rec, frags = reconstruct(frags)
    if rec and mol_to_smiles(rec) == smi:
        # fragments = [Chem.MolToSmiles(frag, isomericSmiles=True, canonical=False) for frag in frags]
        fragments = mols_to_smiles(frags)
        return smi, " ".join(fragments), len(frags)

    return smi, np.nan, 0


def sentence2mol(string, RemoveStereo=False):
    frag_list = string.replace(" ", "").replace("[BOS]", "").replace("[EOS]", "").split('[SEP]')
    frag_list = [frag for frag in frag_list if frag]  # Remove empty strings
    if frag_list == []:
        return None, None
    if len(frag_list[0]) <= 1:
        return None, None
    frag_mol = [Chem.MolFromSmiles(s) for s in frag_list]
    if None in frag_mol:
        return None, None
    mol = reconstruct(frag_mol)[0]
    mol = strip_dummy_atoms(mol)
    if mol is None:
        return None, None
    # Remove chirality information
    if RemoveStereo:
        rdmolops.RemoveStereochemistry(mol)
    smi = Chem.MolToSmiles(mol)
    return mol, smi

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


