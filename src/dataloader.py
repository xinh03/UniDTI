# dataloader.py
# Author  : Ethan Huang
# File    : dataloader.py
# Time    : 2026/3/25

import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from multigrain import MultigrainMolecularFeatures
import dgl
import os
import hashlib

# Drug sequence vocabulary
SMILES_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#()[]=+-*/\\.")
SMILES_TO_IDX = {c: i+1 for i, c in enumerate(SMILES_CHARS)}
UNKNOWN_SMILES_IDX = len(SMILES_TO_IDX) + 1
SMILES_VOCAB_SIZE = UNKNOWN_SMILES_IDX + 1

IUPAC_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()-[]=#+")
IUPAC_TO_IDX = {c: i+1 for i, c in enumerate(IUPAC_CHARS)}
UNKNOWN_IUPAC_IDX = len(IUPAC_TO_IDX) + 1
IUPAC_VOCAB_SIZE = UNKNOWN_IUPAC_IDX + 1

def smiles_to_indices(smiles, max_len=200):
    """Convert SMILES to indices"""
    smiles = str(smiles).strip()
    indices = []
    for char in smiles[:max_len]:
        if char in SMILES_TO_IDX:
            indices.append(SMILES_TO_IDX[char])
        else:
            indices.append(UNKNOWN_SMILES_IDX)
    
    if len(indices) < max_len:
        indices.extend([0] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]
        
    return np.array(indices, dtype=np.int64)

def iupac_to_indices(iupac, max_len=200):
    """Convert IUPAC name to indices"""
    iupac = str(iupac).strip()
    indices = []
    for char in iupac[:max_len]:
        if char in IUPAC_TO_IDX:
            indices.append(IUPAC_TO_IDX[char])
        else:
            indices.append(UNKNOWN_IUPAC_IDX)
    
    if len(indices) < max_len:
        indices.extend([0] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]
        
    return np.array(indices, dtype=np.int64)

def load_contact_map(protein_seq, contact_map_dir, max_prot_len=1200, threshold=0.5):
    """Load or generate protein contact map"""
    try:
        # Generate unique ID from protein sequence
        protein_hash = hashlib.md5(protein_seq.encode()).hexdigest()[:16]
        npy_path = os.path.join(contact_map_dir, f"{protein_hash}.npy")
        
        if os.path.exists(npy_path):
            contact_map = np.load(npy_path)
        else:
            # Return empty map if contact map not found
            contact_map = np.zeros((len(protein_seq), len(protein_seq)), dtype=np.float32)
        
        seq_len = min(contact_map.shape[0], max_prot_len)
        
        # Truncate contact map
        contact_map = contact_map[:seq_len, :seq_len]
        
        # Build DGL graph
        g = dgl.DGLGraph()
        g.add_nodes(seq_len)
        
        # Add edges
        src_nodes = []
        dst_nodes = []
        
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                if contact_map[i, j] > threshold:
                    src_nodes.append(i)
                    dst_nodes.append(j)
        
        if src_nodes:
            g.add_edges(src_nodes, dst_nodes)
            g.add_edges(dst_nodes, src_nodes)  # Undirected graph
            
            # Add edge features
            edge_weights = []
            for i, j in zip(src_nodes, dst_nodes):
                edge_weights.append(contact_map[i, j])
            edge_weights = edge_weights * 2  # Bidirectional edges
            g.edata['weight'] = torch.FloatTensor(edge_weights)
        
        return g
    
    except Exception as e:
        print(f"Failed to load contact map: {e}")
        # Return empty graph
        g = dgl.DGLGraph()
        g.add_nodes(min(len(protein_seq), max_prot_len))
        return g

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290, max_coarse_nodes=50, 
                 max_prot_len=1200, contact_map_dir=None):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.max_coarse_nodes = max_coarse_nodes
        self.max_prot_len = max_prot_len
        self.contact_map_dir = contact_map_dir

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        
        # Multi-grain feature extractor
        self.multigrain_extractor = MultigrainMolecularFeatures()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        smiles = self.df.iloc[index]['SMILES']
        iupac = self.df.iloc[index].get('IUPAC', '')  # If no IUPAC column, use empty string
        
        # 1. Original fine-grained graph (atom-level)
        v_d_graph = self.fc(smiles=smiles, node_featurizer=self.atom_featurizer, 
                           edge_featurizer=self.bond_featurizer)
        
        actual_node_feats = v_d_graph.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d_graph.ndata['h'] = actual_node_feats
        
        if num_virtual_nodes > 0:
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74),
                                           torch.ones(num_virtual_nodes, 1)), 1)
            v_d_graph.add_nodes(num_virtual_nodes, {'h': virtual_node_feat})
        
        v_d_graph = v_d_graph.add_self_loop()
        
        # 2. Multi-grain feature extraction
        multiscale_features = self.multigrain_extractor.extract_multiscale_graph_features(
            smiles, 
            max_atoms=self.max_drug_nodes,
            max_coarse_nodes=self.max_coarse_nodes
        )
        
        if multiscale_features is None:
            multiscale_features = self._get_empty_multiscale_features()
        
        # 3. Drug sequence encoding
        drug_seq_indices = smiles_to_indices(smiles, max_len=200)
        iupac_indices = iupac_to_indices(iupac, max_len=200)
        
        # Add to multiscale features
        multiscale_features['drug_seq_indices'] = drug_seq_indices
        multiscale_features['iupac_indices'] = iupac_indices
        
        # 4. Protein sequence and contact map
        v_p = self.df.iloc[index]['Protein']
        v_p_encoded = integer_label_protein(v_p, max_length=self.max_prot_len)
        y = self.df.iloc[index]['Y']
        
        # Load protein contact graph
        protein_graph = None
        if self.contact_map_dir:
            protein_graph = load_contact_map(v_p, self.contact_map_dir, self.max_prot_len)
        
        return {
            'fine_graph': v_d_graph,
            'multiscale_features': multiscale_features,
            'protein_seq': v_p_encoded,
            'protein_graph': protein_graph,
            'label': y
        }
    
    def _get_empty_multiscale_features(self):
        """Return empty multiscale features"""
        coarse_feat_dim = len(MultigrainMolecularFeatures.COARSE_NODE_TO_IDX) + 10 + 2
        
        return {
            'fine_x': np.zeros((self.max_drug_nodes, 34), dtype=np.float32),
            'fine_edge_index': np.zeros((2, 0), dtype=np.int64),
            'fine_edge_attr': np.zeros((0, 1), dtype=np.float32),
            'coarse_node_features': np.zeros((self.max_coarse_nodes, coarse_feat_dim), dtype=np.float32),
            'coarse_edge_index': np.zeros((2, 0), dtype=np.int64),
            'coarse_edge_attr': np.zeros((0, 1), dtype=np.float32),
            'atom_to_coarse_mapping': np.full((self.max_drug_nodes,), -1, dtype=np.int64),
            'ecfp': np.zeros(1024, dtype=np.float32),
            'descriptors': np.zeros(200, dtype=np.float32),
            'num_atoms': 0,
            'num_coarse_nodes': 0,
            'drug_seq_indices': np.zeros(200, dtype=np.int64),
            'iupac_indices': np.zeros(200, dtype=np.int64)
        }

class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError('n_batches should be > 0')
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders)
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches