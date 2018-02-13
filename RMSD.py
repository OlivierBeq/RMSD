# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

from __future__ import absolute_import, division, print_function

from rdkit import Chem
from collections import defaultdict
import operator
import sys
from bitarray import bitarray

sys.path.append("./atoms_sorter/")

from pythonCanonImplementation import rdkitCanonizer
from pythonCanonImplementation import CanonGraph as cg
from pythonCanonImplementation import RDKit_Graph_Invariants as rdk

LONG_ARRAY = False




    



if __name__ == "__main__":
    mol = Chem.MolFromSmiles("CC1=C(C(=C(C(=C1Cl)Cl)Cl)Cl)Cl")
    frags = Chem.GetMolFrags(mol, asMols=True)

    frag = frags[0]

    cG2 = cg.CanonGraph(frag,
                        rdk.setRDKitAtomNodes,
                        rdk.setRDKitAtomProperties,
                        rdk.setRDKitAtomNeighbors,
                        useEdgeProperties=True,
                        setNeighborsStructuredByEdgeProperties=(
                                rdk.setRDKitAtomNeighborsStructedByBondProperties))

    rdkitCanonizer.canonizeGraph(cG2)

    frag.SetProp("_canonicalRankingNumbers", "True")
    for a in frag.GetAtoms():
        a.SetProp("_canonicalRankingNumber", str(cG2.nodeIndex[a.GetIdx()]))
        print(cG2.nodes[a.GetIdx()])
        print(cG2.nodeIndex[a.GetIdx()])

    frag.UpdatePropertyCache()
    Chem.AddHs(frag);