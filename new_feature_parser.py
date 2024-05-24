# This python file will provide a access point to hack the data_pipeline.process_mmcif function
# Because our mmcif/pdb data can not construct compatible data to the openfold parser.

#Approach 1 use the PDB parser provided with the alphaflow

# Test how to construct protein object with PDB

from openfold.np import protein

# Ask about the relative path on debug

from alphaflow.data.data_pipeline import make_pdb_features, make_protein_features
def create_protein(path):
    with open(path, 'r') as f:
        pdb_string = f.read()
    protein_ob=protein.from_pdb_string(pdb_string)
    return protein_ob

protein_ob = create_protein('/pscratch/sd/l/lemonboy/alphaflow_develop/mmcif_convert/7M7A_A.pdb')

# Works. Potential problems: chain index or the residue_index
# Remind Steph to change the string part

test_feature = make_pdb_features(protein_object=protein_ob,description='', is_distillation=False)

print('debug')

# Works. Try the sequence feature one

protein_feature = make_protein_features(protein_object=protein_ob,description='')

print('debug')


