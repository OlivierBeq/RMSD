# RMSD

Determining the Root Mean Square Deviation (RMSD) between the Maximum Common Substructures (MCS) of two molecules.

The Kuhn-Munkres Hungarian algorithm allows for a fast match of atoms based on:
- atomic symbol
- Sybyl atom types
- pharmacophoric types (i.e. H-bond donors and acceptors or charges)

# Example

```python
from munkres_rmsd import CalcLigRMSD, AtomType
from munkres_rmsd.RMSD import get_example_molecules

# First, load 3D poses of molecules 
mol1, mol2 = get_example_molecules()

# Then compute the RMSD of the best atomic match
rmsd = CalcLigRMSD(mol1, mol2)

print(rmsd) # 10.76150...
```

Let's use Sybyl atom types to match atoms between the two molecules instead of the default, using atomic elements.

```python
# Then compute the RMSD of the best atomic match
rmsd = CalcLigRMSD(mol1, mol2, AtomType.Sybyl)

print(rmsd) # 11.59752...
```

Should you prefer pharmacophore types (i.e. H-bond donors & acceptors, charges and others):

```python
# Then compute the RMSD of the best atomic match
rmsd = CalcLigRMSD(mol1, mol2, AtomType.Pharmacophore)

print(rmsd) # 9.49120...
```
