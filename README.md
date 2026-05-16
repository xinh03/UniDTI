<h1 align="center">
  UniDTI: A Multi-modal and Multi-scale Unified Framework for Predicting Drug-Target Interactions
</h1>
  
<p align="center">
<a href="LINKEDIN_LINK">
<img src="https://img.shields.io/badge/Follow-LinkedIn-0077B5?style=for-the-badge&logo=linkedin" alt="Follow on LinkedIn" />
</a>

<a href="PAPER_LINK">
<img src="https://img.shields.io/badge/Read-Paper-brightgreen?style=for-the-badge" alt="Paper" />
</a>

<a href="LICENSE">
<img src="https://img.shields.io/badge/License-MIT-blueviolet?style=for-the-badge" alt="MIT License" />
</a>

<a href="https://github.com/xinh03/UniDTI/stargazers">
<img src="https://img.shields.io/github/stars/xinh03/UniDTI?style=for-the-badge&color=yellow" alt="GitHub stars" />
</a>
</p>

##  🧠 Abstract
<p align="justify">
   <strong>Abstract:</strong> Accurate identification of drug-target interactions (DTIs) is central to modern drug discovery, yet remains challenging due to the intrinsically multi-scale and heterogeneous nature of molecular recognition. Existing computational approaches typically rely on single-modal or single-scale representations and employ simplistic fusion strategies, limiting their ability to capture the complex, non-linear interplay between ligands and protein binding sites. Here, we present UniDTI, a unified deep learning framework that integrates multi-modal and multi-scale representations of both drugs and proteins within an intent-aware interaction paradigm. Drug molecules are encoded through complementary modalities, including sequence representations (SMILES and IUPAC), hierarchical graph structures (atom-level and substructure-level), and global physicochemical descriptors. Proteins are represented using structure-aware pocket graphs derived from AlphaFold2 models together with sequence-level motif embeddings. To bridge these heterogeneous representations, we introduce a Bidirectional Intention Network (BIN) that models directional and asymmetric interactions between ligand substructures and protein residues, enabling dynamic alignment across modalities and spatial scales beyond conventional attention-based fusion. Across multiple benchmark datasets, including BindingDB, BioSNAP, Davis, DRH, and the GPCR-focused GLASS dataset, UniDTI consistently outperforms state-of-the-art methods in terms of AUROC, AUPRC, and other evaluation metrics, while demonstrating strong generalization ability under cold-start (unseen drug/target) and data-scarce settings. Importantly, to demonstrate its practical utility, we applied UniDTI in a virtual screening against the μ-opioid receptor leading to the experimental identification of two novel active compounds. Overall, UniDTI provides a generalizable and experimentally validated framework for integrating heterogeneous biochemical information, offering a promising direction for next-generation AI-driven drug discovery.
</p>

## 🏛️ Framework
<p align="center">
  <img src="Figure/Fig1.png" width="800">
</p>

## 🛠️ Installation
Clone this Github repo and set up a new conda environment. It normally takes about 20 minutes to install on a normal desktop computer.
1. Clone the source code of UniDTI
   ```bash
   git clone https://github.com/xinh03/UniDTI.git
   cd UniDTI/
   ```
2. Installation Guide
##### 🔹 Method 1: Automatic Creation
  ```bash
  conda env create -f environment.yml
  conda activate unidti
  pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
  pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu121/repo.html
  ```
##### 🔹 Method 2: Step-by-Step Configuration
  ```bash
  # Create and activate the new environment
  conda create -n unidti python=3.9 -y
  conda activate unidti

  # Install Torch, torchvision, and torchaudio with CUDA 12.1 support
  pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

  # Install DGL 2.1.0 from the official DGL CUDA 12.1 repository
  pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu121/repo.html

  # Install DGL-LifeSci for chemoinformatics models
  pip install dgllife==0.3.2

  # Install required helper libraries
  pip install torchdata==0.7.1 pydantic

  # Install RDKit and core scientific libraries
  conda install -c conda-forge rdkit numpy=1.26.3 pandas=1.5.3 scikit-learn=1.4.2 \
  scipy=1.10.1 tqdm prettytable yacs einops networkx=3.2.1 dill packaging \
  matplotlib-base seaborn -y
  ```
3. Activate the environment:
  ```bash
  conda activate unidti
  ```
## 🧮 Data Preparation
##### 🔹 A) Use the processed Prot-GNN data
  ```bash
  cd prot-gnn-data/contact_maps_p2rank
  unzip contact_maps_p2rank.zip
  ```
##### 🔹 B) Preparation for Prot-GNN
  ```bash
  cd prot-gnn-data/
  ```
#### 1. Fetch Protein Structures
P2Rank Usage
  ```bash
  prank predict -f test_data/1fbl.pdb         # predict pockets on single pdb file
  ```
For batch processing, please refer to the official [p2rank](https://github.com/rdk/p2rank)
#### 2. Generate PDB List
  ```bash
  python make_pdb_list.py \
    -i /path/to/pdbs \
    -o /path/to/pdb_list.txt
  ```
#### 3. Protein list pocket prediction
  ```bash
  cd /path/to/p2rank/p2rank_2.5.1 # Please change to your own p2rank directory
  ./prank predict /path/to/your/data/pdb_list.txt -o /path/to/your/data/pdbs-p2rank-results -threads 20 -c alphafold
  ```
#### 4. generate protein contact_map
  ```bash
  env: conda activate unidti
  
  python generate_prot_contact_map.py \
    --pdb_dir /path/to/your/pdbs \
    --p2rank_dir /path/to/your/p2rank_results \
    --output_dir /path/to/your/output_contact_maps \
    --top_k 3 # Optional, default to 3, means keep the first K pockets
  ```

## 🏋️ Training
#### 1. Select task settings
  ```bash
  cd src/
  vim run.sh
  ```
  ```bash
  # Parameter Settings (please modify according to your local setup)
  DATA_NAME="glass"  # options: DAVIS / bindingdb / biosnap / glass / etc.
  SPLIT="warm"  # options: warm / cold_drug / cold_prot / etc.
  
  # Please set the path to your local contact map directory
  CONTACT_MAP_DIR="/path/to/your/contact_maps_p2rank"
  ```
#### 2. Use the input file
  ```bash
  cd datasets/glass/warm
  unzip test.zip
  unzip train.zip
  unzip val.zip
  ```

#### 3. Set the output file path
  ```bash
  vim configs.py
  # Set your output file path
  _C.RESULT.OUTPUT_DIR = "/path/to/your/output"
  ```

#### 4. Start training
  ```bash
  bash run.sh
  ```

## ⚡ Inference
#### 1. set your input file path
  ```bash
  cd src/
  vim prediction.py
  ```
  ```bash
  python prediction.py \
      --test_data /path/to/your/test_data.csv \
      --checkpoint /path/to/your/best_model.pth \
      --result_metrics /path/to/your/result_metrics.pt \
      --output /path/to/save/predictions.csv \
      --contact_map_dir /path/to/your/contact_maps_p2rank \
      --batch_size 64 \
      --device cuda:0
  ```
## 🔍 Repository Structure
```
UniDTI/
├── Figure/ # Figures and visualizations used in the UniDTI
├── datasets/ # Raw and preprocessed datasets
├── models/ # Saved model checkpoints
├── notebook/ # Jupyter notebooks for coarse-grained feature extraction
├── output/ # Output files
├── prot-gnn-data/ # Protein graph data for GNN-based modeling
├── src/ # Source code for feature extraction, training and inference 
├── README.md # Project documentation
└── environment.yml # Conda environment configuration of UniDTI
```
Note: Our code is mainly referenced from [EFGs](https://github.com/rdkit/rdkit/tree/master/Contrib/efgs), M<sup>2</sup>N and [BINDTI](https://github.com/plhhnu/BINDTI).
We gratefully acknowledge the authors for making their code publicly available.  

## 📁 Results
- UniDTI achieves state-of-the-art (SOTA) performance across multiple benchmark datasets, including BindingDB, BIOSNAP, DRH, Davis, and GLASS (GPCR-based).
- Comprehensive experimental evaluations and ablation studies are presented in the manuscript.

## 🧾 Reference
- Ertl, P. An algorithm to identify functional groups in organic molecules. J Cheminform 9, 36 (2017)
- Gonzalo C. EFGs:A Complete and Accurate Implementation of Ertl’s Functional Group Detection Algorithm in RDKit. J.Chem. Inf.Model. 65, 1061−1066 (2025)
- Lv, T. M2N: A Progressive Macro-to-Micro 3D Modeling Scheme for Unveiling Drug-Target Affinity. Proceedings of the AAAI Conference on Artificial Intelligence 39 (1), 586–594 (2025)
- Peng, L. BINDTI: A Bi-Directional Intention Network for Drug-Target Interaction Identification Based on Attention Mechanisms. IEEE Journal of Biomedical and Health Informatics 29 (3), 1602–1612 (2025)


