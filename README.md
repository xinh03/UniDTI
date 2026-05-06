# UniDTI
A Multi-modal and Multi-scale Unified Deep Learning Framework for Drug-Target Interaction Prediction

## 🏗️ Framework
<p align="center">
  <img src="Figure/Fig1.png" width="800">
</p>

## 🛠️ Installation
1. Create the Conda environment:
   ```bash
   conda env create -f environment.yaml
   ```
2. Activate the environment:
   ```bash
   conda activate unidti
   ```
   
## 🏋️ Training
1. Select task settings
  ```bash
  cd src/
  vim run.sh
  ```
  ```bash
  # Parameter Settings (please modify according to your local setup)
  DATA_NAME="GLASS"  # options: DAVIS / BindingDB / BIOSNAP / GLASS
  SPLIT="warm"  # options: warm / cold_drug / cold_prot / etc.
  
  # Please set the path to your local contact map directory
  CONTACT_MAP_DIR="/path/to/your/contact_maps_p2rank"
  ```
2. Set the output file path
  ```bash
  vim configs.py
  _C.RESULT.OUTPUT_DIR = "/path/to/your/output"
  ```
4. Start training
  ```bash
  bash run.sh
  ```


## ⚡ Inference
1. set your input file path
  ```bash
  cd src/
  vim prediction.py
  ```
  ```python
  python test2.2.py \
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
└── environment.yaml # Conda environment configuration of UniDTI
```
Note: Our code is mainly referenced from [EFGs](https://github.com/rdkit/rdkit/tree/master/Contrib/efgs) and [BINDTI](https://github.com/plhhnu/BINDTI).
We gratefully acknowledge the authors for making their code publicly available.  


## 📁 Results
UniDTI consistently outperforms baseline methods across multiple benchmark datasets (BindingDB, BioSNAP, DRH, Davis and Glass(GPCR-Based)).
Detailed experimental results and ablation studies are reported in the manuscript.

## 🧾 Reference
Ertl, P. An algorithm to identify functional groups in organic molecules. J Cheminform 9, 36 (2017)
Gonzalo C. EFGs:AComplete andAccurate Implementation ofErtl’sFunctional Group Detection Algorithm inRDKit. J.Chem. Inf.Model. 65, 1061−1066 (2025)
Lihong P. BINDTI: A Bi-Directional Intention Network for Drug-Target Interaction Identification Based on Attention Mechanisms. IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 3, MARCH 2025

