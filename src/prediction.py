# prediction.py
# Author  : Ethan Huang
# File    : prediction.py
# Time    : 2026/3/25

"""
#env:
#seed 42
python prediction.py \
    --test_data /home/chuym/mor/dl/Uni-DTA_5.2_prot_gnn/data/glass-100-700da/warm/test.csv \
    --checkpoint /home/chuym/mor/dl/BINDTI-UNIDTI/BINDTI/BINDTI/output/result/multimodal-multigrain2/glass-100-700da/seed42/best_model_epoch_45.pth \
    --result_metrics /home/chuym/mor/dl/BINDTI-UNIDTI/BINDTI/BINDTI/output/result/multimodal-multigrain2/glass-100-700da/seed42/result_metrics.pt \
    --output /home/chuym/mor/dl/BINDTI-UNIDTI/BINDTI/BINDTI/output/test/UNIDTI/GLASS/seed42/predictions.csv \
    --contact_map_dir /home/shaoxin/uni-dta/BINDTI/BINDTI/prot-gnn-data/glass-100-700da/contact_maps_p2rank \
    --batch_size 64 \
    --device cuda:0
"""

import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from time import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import UniDTI
from dataloader import DTIDataset
from configs import get_cfg_defaults
from utils import set_seed
import dgl

def multimodal_collate_fn(batch):
    """Batch collation for multimodal data."""
    fine_graphs = [item['fine_graph'] for item in batch]
    multiscale_features = [item['multiscale_features'] for item in batch]
    protein_seqs = torch.stack([torch.tensor(item['protein_seq']) for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    
    from dgl import batch as dgl_batch
    batched_fine_graph = dgl_batch(fine_graphs)
    
    batched_multiscale = {}
    keys = multiscale_features[0].keys()
    for key in keys:
        feats_list = [item[key] for item in multiscale_features]
        if isinstance(feats_list[0], np.ndarray):
            ndim = feats_list[0].ndim
            max_shape = [max(f.shape[d] for f in feats_list) for d in range(ndim)]
            padded_batch = np.zeros((len(feats_list), *max_shape), dtype=feats_list[0].dtype)
            for i, feat in enumerate(feats_list):
                idx = tuple([i] + [slice(0, s) for s in feat.shape])
                padded_batch[idx] = feat
            batched_multiscale[key] = torch.tensor(padded_batch)
        else:
            batched_multiscale[key] = torch.tensor(feats_list)
    
    drug_seq_indices = torch.stack([torch.tensor(item['drug_seq_indices']) for item in multiscale_features])
    iupac_indices = torch.stack([torch.tensor(item['iupac_indices']) for item in multiscale_features])
    
    prot_graphs = [item['protein_graph'] for item in batch]
    valid_prot_graphs = [g for g in prot_graphs if g is not None]
    if valid_prot_graphs:
        batched_prot_graph = dgl_batch(valid_prot_graphs)
        batch_ids = torch.cat([
            torch.full((g.num_nodes(),), i, dtype=torch.long)
            for i, g in enumerate(valid_prot_graphs)
        ])
        batched_prot_graph.ndata['batch_id'] = batch_ids
    else:
        batched_prot_graph = None
    
    return {
        'fine_graph': batched_fine_graph,
        'multiscale_features': batched_multiscale,
        'drug_seq': drug_seq_indices,
        'iupac_seq': iupac_indices,
        'protein_seq': protein_seqs,
        'protein_graph': batched_prot_graph,
        'label': labels
    }

def predict(model, dataloader, device):
    """Predict probabilities and return labels and probs."""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            v_d = batch_data['fine_graph'].to(device)
            v_p = batch_data['protein_seq'].to(device)
            labels = batch_data['label'].float().to(device)
            
            multiscale_features = batch_data['multiscale_features']
            drug_seq = batch_data.get('drug_seq', None)
            iupac_seq = batch_data.get('iupac_seq', None)
            protein_graph = batch_data.get('protein_graph', None)
            
            if protein_graph is not None:
                protein_graph = protein_graph.to(device)
            if drug_seq is not None:
                drug_seq = drug_seq.to(device)
            if iupac_seq is not None:
                iupac_seq = iupac_seq.to(device)
            
            _, _, score, _ = model(v_d, v_p, multiscale_features, drug_seq, iupac_seq, protein_graph, mode="eval")
            
            probs = torch.sigmoid(score).squeeze()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs)

def main():
    parser = argparse.ArgumentParser(description="Test UniDTI model and output predictions")
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test CSV file (must contain columns: SMILES, IUPAC, Protein, Y)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--result_metrics', type=str, required=True,
                        help='Path to result_metrics.pt file (containing training config)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path (default: predictions.csv)')
    parser.add_argument('--contact_map_dir', type=str, default='../prot-gnn-data/glass/contact_maps',
                        help='Directory containing protein contact maps')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing (default: 64)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0 or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading config from {args.result_metrics}")
    result_state = torch.load(args.result_metrics, map_location='cpu')
    config = result_state['config']
    print("Config loaded.")

    print("Building model...")
    model = UniDTI(device=device, **config).to(device)
    print(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    print(f"Loading test data from {args.test_data}")
    df_test = pd.read_csv(args.test_data)
    
    max_drug_nodes = config.get("DRUG", {}).get("MAX_NODES", 290)
    max_coarse_nodes = config.get("DRUG", {}).get("MULTIGRAIN", {}).get("MAX_COARSE_NODES", 50)
    max_prot_len = config.get("PROTEIN", {}).get("MAX_LEN", 1200)
    
    test_dataset = DTIDataset(
        df_test.index.values,
        df_test,
        max_drug_nodes=max_drug_nodes,
        max_coarse_nodes=max_coarse_nodes,
        max_prot_len=max_prot_len,
        contact_map_dir=args.contact_map_dir
    )
    print(f"Test dataset size: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=multimodal_collate_fn
    )

    print("Running prediction...")
    labels, probs = predict(model, test_loader, device)

    # ========== Output full CSV ==========
    out_df = df_test.copy()
    out_df['probability'] = probs
    out_df['pred_label'] = (probs > 0.5).astype(int)
    out_df.to_csv(args.output, index=False)
    print(f"Complete predictions saved to {args.output} (contains original columns + probability + pred_label)")
    # ========================================

if __name__ == '__main__':
    print(f"Start time: {datetime.now()}")
    start = time()
    main()
    end = time()
    print(f"End time: {datetime.now()}")
    print(f"Total running time: {round(end - start, 2)}s")