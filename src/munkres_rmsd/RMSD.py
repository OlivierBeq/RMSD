# -*- coding: utf-8 -*-

import lzma
from enum import Enum, auto
from typing import Optional, Union, Tuple

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


class AtomType(Enum):
    Symbol = auto()
    Sybyl = auto()
    Pharmacophore = auto()
    Skeleton = auto()


def _sybyl_atom_type(atom: Chem.Atom) -> str:
    """Assign sybyl atom type.

    From the Open Drug Discovery Toolkit (ODDT; https://github.com/oddt/oddt)
    developed by Maciej Wójcikowski

    LICENSE

    Copyright (c) 2014, Maciej Wójcikowski
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the {organization} nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Reference #1: http://www.tripos.com/mol2/atom_types.html
    Reference #2: http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf

    :param atom: RDKit atom
    """
    sybyl = None
    atom_symbol = atom.GetSymbol()
    atomic_num = atom.GetAtomicNum()
    hyb = atom.GetHybridization() - 1  # -1 since 1 = sp, 2 = sp1 etc
    hyb = min(hyb, 3)
    degree = atom.GetDegree()
    aromtic = atom.GetIsAromatic()

    # define groups for atom types
    guanidine = '[NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])'  # strict
    # guanidine = '[NX3]([!O])([!O])!:C!:[NX3]([!O])([!O])' # corina compatible
    # guanidine = '[NX3]!@C(!@[NX3])!@[NX3,NX2]'
    # guanidine = '[NX3]C([NX3])=[NX2]'
    # guanidine = '[NX3H1,NX2,NX3H2]C(=[NH1])[NH2]' # previous
    #

    if atomic_num == 6:
        if aromtic:
            sybyl = 'C.ar'
        elif degree == 3 and _atom_matches_smarts(atom, guanidine):
            sybyl = 'C.cat'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 7:
        if aromtic:
            sybyl = 'N.ar'
        elif _atom_matches_smarts(atom, 'C(=[O,S])-N'):
            sybyl = 'N.am'
        elif degree == 3 and _atom_matches_smarts(atom, '[$(N!-*),$([NX3H1]-*!-*)]'):
            sybyl = 'N.pl3'
        elif _atom_matches_smarts(atom, guanidine):  # guanidine has N.pl3
            sybyl = 'N.pl3'
        elif degree == 4 or hyb == 3 and atom.GetFormalCharge():
            sybyl = 'N.4'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 8:
        # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        if degree == 1 and _atom_matches_smarts(atom, '[CX3](=O)[OX1H0-]'):
            sybyl = 'O.co2'
        elif degree == 2 and not aromtic:  # Aromatic Os are sp2
            sybyl = 'O.3'
        else:
            sybyl = 'O.2'
    elif atomic_num == 16:
        # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        if degree == 3 and _atom_matches_smarts(atom, '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]'):
            sybyl = 'S.O'
        # https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
        elif _atom_matches_smarts(atom, 'S(=,-[OX1;+0,-1])(=,-[OX1;+0,-1])(-[#6])-[#6]'):
            sybyl = 'S.o2'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 15 and hyb == 3:
        sybyl = '%s.%i' % (atom_symbol, hyb)
    if not sybyl:
        sybyl = atom_symbol
    return sybyl


def _pharmacophoric_atom_type(atom: Chem.Atom) -> str:
    """Assign pharmacophoric atom type (H-bond donor, Halogen, ...).

    :param atom: RDKit atom
    """
    charge = atom.GetFormalCharge()
    if _atom_matches_smarts(atom, '[O,N;!H0]'):
        return 'HBD'
    elif _atom_matches_smarts(atom, '[C,N;R0]=O'):
        return 'HBA'
    elif charge > 0:
        return f'+{charge}'
    elif charge < 0:
        return f'-{charge}'
    else:
        return 'ANY'


def _atom_matches_smarts(atom, smarts):
    """Find if the atom is part of its molecule's match with the provided SMARTS.

    From the Open Drug Discovery Toolkit (ODDT; https://github.com/oddt/oddt)
    developed by Maciej Wójcikowski

    LICENSE

    Copyright (c) 2014, Maciej Wójcikowski
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the {organization} nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Reference #1: http://www.tripos.com/mol2/atom_types.html
    Reference #2: http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf

    :param atom: RDKit atom
    :param smarts: SMARTS pattern to match
    """
    idx = atom.GetIdx()
    patt = Chem.MolFromSmarts(smarts)
    for m in atom.GetOwningMol().GetSubstructMatches(patt):
        if idx in m:
            return True
    return False


def atom_match(atom1, atom2, atom_types):
    if atom_types is AtomType.Symbol:
        return atom1.GetSymbol() == atom2.GetSymbol()
    elif atom_types is AtomType.Sybyl:
        return _sybyl_atom_type(atom1) == _sybyl_atom_type(atom2)
    elif atom_types is AtomType.Pharmacophore:
        return _pharmacophoric_atom_type(atom1) == _pharmacophoric_atom_type(atom2)
    elif atom_types is AtomType.Skeleton:
        return True  # Any atom matches any other
    else:
        raise NotImplementedError(f'AtomType {atom_types} is not implemented')


def CalcLigRMSD(lig1: Chem.Mol,
                lig2: Chem.Mol,
                atom_types: AtomType = AtomType.Symbol,
                lig1_conf: int = -1,
                lig2_conf: int = -1,
                return_corr: Optional[bool] = False
                ) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """Align two ligands and obtain the Root-mean-square deviation (RMSD) between them.

    The Kuhn-Munkres (Hungarian) algorithm matches atoms of one ligand to the other based on the distance between them.
    if one ligand structure has missing atoms (e.g. undefined electron density in the crystal structure),
    the RMSD is calculated for the maximum common substructure (MCS).

    Adapted from https://doi.org/10.1021/ci400534h

    :param lig1: RDKit molecule
    :param lig2: RDKit molecule
    :param atom_types: Types of atomic matches to be allowed.
    :param lig1_conf: Index of the conformer of lig1 to consider
    :param lig2_conf: Index of the conformer of lig2 to consider
    :param return_corr: True to return the correspondence between atoms of lig1 and lig2
    :return: Root-mean-square deviation between the two input molecules
    """
    # Exclude hydrogen atoms from the calculation
    lig1 = Chem.AddHs(lig1)
    lig2 = Chem.AddHs(lig2)
    # Get conformers
    conf1 = lig1.GetConformer(lig1_conf)
    conf2 = lig2.GetConformer(lig2_conf)
    # Define cost matrix
    cost = np.zeros((lig1.GetNumAtoms(), lig2.GetNumAtoms()))
    # Iterate only on the minimum number of atoms of the 2 molecules
    dims = min(cost.shape[0], cost.shape[1])
    # Fill in the cost matrix
    for i in range(dims):
        for j in range(dims):
            x, y = lig1.GetAtomWithIdx(i), lig2.GetAtomWithIdx(j)
            if atom_match(x, y, atom_types):  # Atoms are similar
                # Cost is the squared distance between the 2 atoms
                cost[i, j] = conf1.GetAtomPosition(i).Distance(conf2.GetAtomPosition(j)) ** 2
            else:  # Atoms do not match
                cost[i, j] = float('inf')
    # Solve the assignment for minimum total distance
    try:
        opt_lig1, opt_lig2 = linear_sum_assignment(cost, maximize=False)
    except ValueError as e:
        raise ValueError('Could not find atomic matches.') from e
    rmsd = np.sqrt(cost[opt_lig1, opt_lig2].sum())
    # Return
    if return_corr:
        return rmsd, opt_lig1, opt_lig2
    return rmsd

def get_example_molecules():
    """Obtain one of the two example molecules."""
    mol1 = b"\xfd7zXZ\x00\x00\x04\xe6\xd6\xb4F\x02\x00!\x01\x16\x00\x00\x00t/\xe5\xa3\xe0\x03\x9f\x00\xef]\x00\x05\x08\xf8/\xcf\x81F\xe9\xa5b\xde\xd3\x00R\xfafx\x8a\xe7IF\x97\xb3\x9a\xb9\x1b<LB\xefG\x89rN\xd9lAKqe^ \xfc\xa1\x9e\xf9$ \x88\x1dL\x9c\x05\x7f)\xfb\xee@e^\xb7\x8a\x98\xe6n\x89\xaf\xc7\x05>\xab-9\x1f\xac\xc6\xaa\xeeD\x88\x15\xe0X\x99\x9f\xcf{B]0\x0e\x95L\x87\xcc\xf3\xbf\x8c\x0cIDYG\xed\x8ed%M\xee\x8a\x86\xec\xaeE\xa4h\x01\xe4\xfaG\xec\x860\x8f\xb3\x97kC\xfb\x04\x17V\xf4\xe2\xf1T\xbf$\x86\xdd'\xf0\xaa\n\x9a\x1a\xfc$\xd3\x9az\xc35l\xeb\xfeZi\x12\xbd\xcf\xa6)\xd6k$M\xe52;\x8a\xff2b\xa9\xbf\xa5)\\L:\x96\xe4\x998\xa5\xdelj\x1baJ\x1eK\x1f\x0cU\xe2z8\x06\xf0/_ m\x1a\x99a\xc5\x92&3J\x0f\r\xc7Ax&\xfbxR\x8aGr;\x07\xc3 ^\xe0\xa4 \xc9\xe4\x98\x08\xc0\x00\x00BX\x82\xef\xc1\xc5\x94\xad\x00\x01\x8b\x02\xa0\x07\x00\x00&l\xc6\x85\xb1\xc4g\xfb\x02\x00\x00\x00\x00\x04YZ"
    mol2 = b'\xfd7zXZ\x00\x00\x04\xe6\xd6\xb4F\x02\x00!\x01\x16\x00\x00\x00t/\xe5\xa3\xe0\x04E\x01\x12]\x00\x05\x08\xf8/\xcf\x81F\xe9\xa5b\xde\xd3\x00R\xfafx\x8b\xf3(R\xf5\x91\xda\x01\xcc\xa2BD\xedR2\xc3\xbeQ\x01\x16\xfdl\xf9\xae\xe1r\xda:~\x06\x91\xb5\x12\xee\xfaD\xffhh\xb2M\x08\x94\x9e\x04\xa4\xd4\x0b\xdd\xd3_1\x93\x83\x1a@E\xeb\xe5wS\xda\x95\xafwp\x03\xd51+\x94\xc2qO\xf3n\x08\xeb\x83\xe1]E\xd7\x8e(N\xe4\xbb\x1c\xf6$\xb8\xf5\xd6\x93\xa6\xa8\x87\xc8\xf6\xe2.]\x18\xb5\xf3\xe6k\x8f\x15"\xd5YT\x85\xccu^R\x9f\x99\xb6\xbc \x10\x84\x8d\x83\x03\xb0\xd4\xcb\xea\x14\xc5a\xfcb\x00y\xb4{U\xc9\xeb>\xfa)\x00\xdd\x93\xb4\xaa\x0cjM\r\x02\xf9V\xb9POU\x9d\xea\x1e\xfa\x88\xed!3\x8b"O\xcaM`\x13\xc6\x84>\x006\xa6\xa4`\xcfh\xc4p\xc8\xeeZ\x90z\xc6\xb5\xc6\x10\xcb\x865\x19\xecvj\x88{\xd3\x1d\xe8\xc9\xddI<\xa2\x02\xf4e\xa3\x01,\xd1l\xfb\xe5\xa6\x1f\x7f\xbd\x9f)\x140P;\xf1\xb1\xc0\xfb|\xc0\x8d\xc3\xfb\x1b\xfd\xca\\\x0f\x15\xd6]\xd9\x0f \x00\x00\x00\xd0z*5\x9d\x1f\xb0\x90\x00\x01\xae\x02\xc6\x08\x00\x00A\x17\xb6\xc7\xb1\xc4g\xfb\x02\x00\x00\x00\x00\x04YZ'
    mol1, mol2 = lzma.decompress(mol1).decode(), lzma.decompress(mol2).decode()
    mol1, mol2 = Chem.MolFromMolBlock(mol1), Chem.MolFromMolBlock(mol2)
    return mol1, mol2
