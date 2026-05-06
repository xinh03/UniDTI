# get_p2rank_prediction.sh
# Author  : Ethan Huang
# File    : get_p2rank_prediction.sh
# Time    : 2026/3/25

# Single protein pocket prediction
# ./prank predict -f /path/to/your/data/glass/pdbs/A5D7K8.pdb -o /path/to/your/data/glass/pdbs-p2rank-test -c alphafold

# Protein list pocket prediction
# conda deactivate
# conda activate p2rank

cd /path/to/p2rank/p2rank_2.5.1 # Please change to your own p2rank directory
./prank predict /path/to/your/data/pdb_list.txt -o /path/to/your/data/pdbs-p2rank-results -threads 20 -c alphafold
