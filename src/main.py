# main.py
# Author  : Ethan Huang
# File    : main.py
# Time    : 2026/3/25

from models import UniDTI
from time import time
from utils import set_seed, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="UniDTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='sample')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", 
                    choices=['warm', 'cold_drug','cold_prot'])
parser.add_argument('--contact_map_dir', type=str, default='../prot-gnn-data/glass/contact_maps',
                    help='Contact map directory path')
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR + f'{args.data}/{args.split}')

    print("start...")
    print(f"dataset:{args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'../datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train, 
                              max_drug_nodes=cfg.DRUG.MAX_NODES,
                              max_coarse_nodes=cfg.DRUG.MULTIGRAIN.MAX_COARSE_NODES,
                              max_prot_len=cfg.PROTEIN.MAX_LEN,
                              contact_map_dir=args.contact_map_dir)
    print(f'train_dataset:{len(train_dataset)}')
    val_dataset = DTIDataset(df_val.index.values, df_val,
                            max_drug_nodes=cfg.DRUG.MAX_NODES,
                            max_coarse_nodes=cfg.DRUG.MULTIGRAIN.MAX_COARSE_NODES,
                            max_prot_len=cfg.PROTEIN.MAX_LEN,
                            contact_map_dir=args.contact_map_dir)
    test_dataset = DTIDataset(df_test.index.values, df_test,
                             max_drug_nodes=cfg.DRUG.MAX_NODES,
                             max_coarse_nodes=cfg.DRUG.MULTIGRAIN.MAX_COARSE_NODES,
                             max_prot_len=cfg.PROTEIN.MAX_LEN,
                             contact_map_dir=args.contact_map_dir)

    def multimodal_collate_fn(batch):
        """Multimodal data batch collate function"""
        fine_graphs = [item['fine_graph'] for item in batch]
        multiscale_features = [item['multiscale_features'] for item in batch]
        protein_seqs = torch.stack([torch.tensor(item['protein_seq']) for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        
        # Batch fine graphs using DGL
        from dgl import batch as dgl_batch
        batched_fine_graph = dgl_batch(fine_graphs)
        
        # Batch multiscale features with padding
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
        
        # Extract sequence indices from multiscale features
        drug_seq_indices = torch.stack([torch.tensor(item['drug_seq_indices']) for item in multiscale_features])
        iupac_indices = torch.stack([torch.tensor(item['iupac_indices']) for item in multiscale_features])
        
        # Extract protein graph data
        prot_graphs = [item['protein_graph'] for item in batch]
        valid_prot_graphs = [g for g in prot_graphs if g is not None]
        if valid_prot_graphs:
            batched_prot_graph = dgl_batch(valid_prot_graphs)
            # Add batch_id attribute to each node for subgraph separation
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

    params = {
        'batch_size': cfg.SOLVER.BATCH_SIZE, 
        'shuffle': True, 
        'num_workers': cfg.SOLVER.NUM_WORKERS,
        'drop_last': True, 
        'collate_fn': multimodal_collate_fn
    }

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = UniDTI(device=device, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, 
                     args.data, args.split, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))

    print(f"\nDirectory for saving result: {cfg.RESULT.OUTPUT_DIR}{args.data}")
    print(f'\nend...')

    return result

if __name__ == '__main__':
    print(f"start time: {datetime.now()}")
    s = time()
    result = main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s")