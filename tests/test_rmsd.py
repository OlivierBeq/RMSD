
import os
import unittest
from rdkit import Chem
from rdkit.Chem import AllChem

from munkres_rmsd import CalcLigRMSD, AtomType


class TestRMSDMethods(unittest.TestCase):
    """Test API exposure."""

    def setUp(self) -> None:
        """Set up the test case."""
        # Input molecules
        with Chem.SDMolSupplier(os.path.join(os.path.dirname(__file__), 'resources', 'SD006053.sdf')) as supplier:
            self.mol1 = next(supplier)
        with Chem.SDMolSupplier(os.path.join(os.path.dirname(__file__), 'resources', 'SD006054.sdf')) as supplier:
            self.mol2 = next(supplier)
        # Embedding technique
        self.params = AllChem.KDG()
        self.params.maxIterations = 10_000
        self.params.useRandomCoords = True
        self.params.randomSeed = 1


    def test_rmsd_1(self):
        mol1 = Chem.MolFromSmiles("C1=CNC=CN1")
        mol2 = Chem.MolFromSmiles("C1=CC=C(C=C1)N")
        mol1 = Chem.AddHs(mol1)
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol1, self.params)
        AllChem.EmbedMolecule(mol2, self.params)
        rmsd, assignment_l1, assignment_l2 = CalcLigRMSD(mol1, mol2, AtomType.Skeleton, return_corr=True)
        self.assertIsNotNone(assignment_l1)
        self.assertIsNotNone(assignment_l2)
        self.assertIsNotNone(rmsd)
        self.assertAlmostEqual(rmsd, 2.729819002443781)

    def test_rmsd_2(self):
        mol1 = Chem.MolFromSmiles("C1=CNC=CN1")
        mol2 = Chem.MolFromSmiles("C1=CC=C(C=C1)N")
        mol1 = Chem.AddHs(mol1)
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol1, self.params)
        AllChem.EmbedMolecule(mol2, self.params)
        with self.assertRaises(ValueError):
            CalcLigRMSD(mol1, mol2, AtomType.Symbol, return_corr=False)

    def test_rmsd_3(self):
        rmsd = CalcLigRMSD(self.mol1, self.mol1, AtomType.Skeleton)
        self.assertAlmostEqual(rmsd, 0)
    
    def test_rmsd_4(self):
        rmsd = CalcLigRMSD(self.mol1, self.mol2, AtomType.Skeleton)
        self.assertAlmostEqual(rmsd, 0)

    def test_rmsd_5(self):
        mol = Chem.MolFromSmiles("CC(C)CC(C(=O)NC(CC(=O)N)C(=O)NC(CC1=CC=CC=C1)"
                                 "C(=O)NC(C(C)O)C(=O)NCC(=O)NC(CCSC)C(=O)N2CCCC"
                                 "2C(=O)N3CCCC3C(=O)NC(C)C(=O)NC(CC(=O)O)C(=O)N"
                                 "C(CCC(=O)O)C(=O)NC(CC(=O)O)C(=O)NC(CC4=CC=C(C"
                                 "=C4)O)C(=O)NC(CO)C(=O)N5CCCC5C(=O)N)NC(=O)C(C"
                                 "C(=O)O)NC(=O)C(CC(=O)O)NC(=O)C(CC6=CC=CC=C6)N"
                                 "C(=O)C(CC(=O)O)NC(=O)C(CC7=CNC8=CC=CC=C87)NC("
                                 "=O)C(CCSC)NC(=O)C")
        mol_rdm = Chem.MolFromSmiles(Chem.MolToSmiles(mol, doRandom=True))
        mol = Chem.AddHs(mol)
        mol_rdm = Chem.AddHs(mol_rdm)
        AllChem.EmbedMolecule(mol, self.params)
        AllChem.EmbedMolecule(mol_rdm, self.params)
        rmsd, assignment_l1, assignment_l2 = CalcLigRMSD(mol, mol_rdm, AtomType.Symbol, return_corr=True)
        self.assertTrue(all([mol.GetAtoms()[x].GetSymbol() == mol_rdm.GetAtoms()[y].GetSymbol()
                             for x, y in zip(assignment_l1.tolist(), assignment_l2.tolist())]))


if __name__ == '__main__':
    unittest.main()
