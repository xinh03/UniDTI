# models.py
# Author  : Ethan Huang
# File    : models.py
# Time    : 2026/3/25

import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from ACmix import ACmix
from BIN import BiIntention
import dgl
import numpy as np

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

class SequenceEncoder(nn.Module):
    """Sequence encoder (BiLSTM)"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3, use_bilstm=True, max_len=200):
        super(SequenceEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_bilstm = use_bilstm
        self.max_len = max_len

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Positional embedding, dimension matches embedding_dim
        self.position_embedding = nn.Embedding(max_len, embedding_dim)

        # LSTM layer
        lstm_input_dim = embedding_dim
        lstm_hidden_dim = hidden_dim // 2 if use_bilstm else hidden_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=use_bilstm,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        # Output layer normalization (hidden_dim is the full dimension of LSTM output)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Clamp indices to valid range
        x = torch.clamp(x, 0, self.vocab_size - 1)

        batch_size, seq_len = x.size()

        # Word embeddings
        x_embed = self.embedding(x)  # [batch, seq_len, embedding_dim]

        # Positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.position_embedding(positions)  # [batch, seq_len, embedding_dim]

        # Add and dropout
        x_embed = x_embed + pos_embed
        x_embed = self.dropout(x_embed)

        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x_embed)

        if self.use_bilstm:
            # Concatenate final hidden states from both directions -> [batch, hidden_dim]
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]  # [batch, hidden_dim]

        # Global average pooling
        lstm_pooled = torch.mean(lstm_out, dim=1)
        lstm_pooled = self.norm(lstm_pooled)

        return lstm_pooled, hidden

class DrugSequenceEncoder(nn.Module):
    """Drug multi-sequence encoder (SMILES + IUPAC)"""
    def __init__(self, smiles_vocab_size=68, iupac_vocab_size=66,
                 embedding_dim=64, hidden_dim=128, num_layers=2,
                 dropout=0.3, use_bilstm=True):
        super(DrugSequenceEncoder, self).__init__()
        self.smiles_vocab_size = smiles_vocab_size
        self.iupac_vocab_size = iupac_vocab_size

        # SMILES encoder
        self.smiles_encoder = SequenceEncoder(
            vocab_size=smiles_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_bilstm=use_bilstm
        )

        # IUPAC encoder
        self.iupac_encoder = SequenceEncoder(
            vocab_size=iupac_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_bilstm=use_bilstm
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.hidden_dim = hidden_dim

    def forward(self, smiles_indices, iupac_indices):
        # Clamp indices
        smiles_indices = torch.clamp(smiles_indices, 0, self.smiles_vocab_size - 1)
        iupac_indices = torch.clamp(iupac_indices, 0, self.iupac_vocab_size - 1)

        smiles_pooled, _ = self.smiles_encoder(smiles_indices)
        iupac_pooled, _ = self.iupac_encoder(iupac_indices)

        seq_features = torch.cat([smiles_pooled, iupac_pooled], dim=-1)
        seq_features = self.fusion(seq_features)

        return seq_features

class ProteinGCN(nn.Module):
    """Protein sparse residue GNN encoder (using DGL)"""
    def __init__(self, in_feats, hidden_dim=128, num_layers=3, dropout=0.3):
        super(ProteinGCN, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GCN layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            dgl.nn.GraphConv(in_feats, hidden_dim, activation=F.relu)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                dgl.nn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
            )
        
        # Output layer
        self.layers.append(dgl.nn.GraphConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, g, features):
        h = features
        
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = self.dropout(h)
        
        # Global pooling
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
        
        hg = self.norm(hg)
        return hg, h

class EnhancedProteinEncoder(nn.Module):
    """Enhanced protein encoder (ACmix + GCN)"""
    def __init__(self, embedding_dim, num_filters, num_head, padding=True,
                 gnn_hidden_dim=128, gnn_layers=3, gnn_dropout=0.3):
        super(EnhancedProteinEncoder, self).__init__()
        
        # Original ACmix encoder
        self.acmix_encoder = ProteinACmix(embedding_dim, num_filters, num_head, padding)
        
        # GNN encoder
        self.gnn = ProteinGCN(
            in_feats=num_filters[-1],
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            dropout=gnn_dropout
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(num_filters[-1] + gnn_hidden_dim, num_filters[-1]),
            nn.LayerNorm(num_filters[-1]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, v, protein_graph=None):
        # ACmix encoding
        acmix_features = self.acmix_encoder(v)  # [batch, seq_len, hidden]

        if protein_graph is not None and hasattr(protein_graph.ndata, 'batch_id'):
            batch_size = v.size(0)
            gnn_features_list = []
            
            for i in range(batch_size):
                # Extract subgraph
                mask = protein_graph.ndata['batch_id'] == i
                subgraph = dgl.node_subgraph(protein_graph, mask)
                
                if subgraph.num_nodes() > 0:
                    # Get ACmix features for the corresponding positions (only actual nodes)
                    sub_features = acmix_features[i:i+1, :subgraph.num_nodes(), :]
                    # GNN encoding
                    gnn_feat, _ = self.gnn(subgraph, sub_features.squeeze(0))
                    gnn_features_list.append(gnn_feat)
                else:
                    gnn_features_list.append(torch.zeros(1, self.gnn.hidden_dim, device=v.device))
            
            # Stack GNN features for all proteins
            gnn_features = torch.cat(gnn_features_list, dim=0)
            
            # Fuse ACmix and GNN features
            acmix_global = torch.mean(acmix_features, dim=1)
            fused_features = torch.cat([acmix_global, gnn_features], dim=-1)
            fused_features = self.fusion(fused_features)
            
            # Expand back to sequence dimension (residual connection)
            fused_features = fused_features.unsqueeze(1).expand(-1, acmix_features.size(1), -1)
            output_features = acmix_features + 0.1 * fused_features
        else:
            output_features = acmix_features

        return output_features

class MultigrainMolecularEncoder(nn.Module):
    """Enhanced multi-grain molecular encoder (integrating sequence information)"""
    
    def __init__(self, atom_dim=34, coarse_node_dim=15, ecfp_dim=1024, desc_dim=200,
                 hidden_dim=128, max_atoms=150, max_coarse_nodes=50, 
                 cross_scale_interaction=True, use_global_prop=True, 
                 device='cuda', smiles_vocab_size=68, iupac_vocab_size=66,
                 use_coarse=True):  ## New parameter to control use of coarse-grained features
        super(MultigrainMolecularEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        self.max_coarse_nodes = max_coarse_nodes
        self.cross_scale_interaction = cross_scale_interaction
        self.use_global_prop = use_global_prop
        self.use_coarse = use_coarse  ##
        self.device = device
        
        # Fine-grained atom encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Coarse-grained node encoder
        self.coarse_encoder = nn.Sequential(
            nn.Linear(coarse_node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Drug sequence encoder
        self.drug_seq_encoder = DrugSequenceEncoder(
            smiles_vocab_size=smiles_vocab_size,
            iupac_vocab_size=iupac_vocab_size,
            embedding_dim=hidden_dim // 2,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.3,
            use_bilstm=True
        )
        
        # Global property encoder
        if use_global_prop:
            self.ecfp_encoder = nn.Sequential(
                nn.Linear(ecfp_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            self.desc_encoder = nn.Sequential(
                nn.Linear(desc_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            self.global_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # Including sequence features
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        else:
            self.global_fusion = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  # Only sequence features
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        
        # Cross-scale interaction module
        if cross_scale_interaction:
            self.cross_scale_fusion = CrossScaleFusion(
                hidden_dim=hidden_dim,
                dropout=0.3,
                device=device
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, multiscale_features, smiles_indices=None, iupac_indices=None):
        batch_size = multiscale_features['fine_x'].shape[0]
        
        # 1. Encode fine-grained atom features
        fine_x = multiscale_features['fine_x'].to(self.device)
        fine_emb = self.atom_encoder(fine_x)
        
        # 2. Encode coarse-grained node features (new: executed based on use_coarse)
        if self.use_coarse:
            coarse_features = multiscale_features['coarse_node_features'].to(self.device)
            coarse_emb = self.coarse_encoder(coarse_features)
        else:
            coarse_emb = None  # Skip coarse-grained when disabled
        
        # 3. Encode sequence features
        if smiles_indices is not None and iupac_indices is not None:
            seq_emb = self.drug_seq_encoder(smiles_indices, iupac_indices)
        else:
            seq_emb = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
        # 4. Encode global properties (based on use_global_prop)
        if self.use_global_prop:
            ecfp = multiscale_features['ecfp'].to(self.device)
            desc = multiscale_features['descriptors'].to(self.device)
            
            ecfp_emb = self.ecfp_encoder(ecfp)
            desc_emb = self.desc_encoder(desc)
            
            # Fuse all global features: ECFP, descriptors, sequence
            global_emb = torch.cat([ecfp_emb, desc_emb, seq_emb], dim=-1)
        else:
            global_emb = seq_emb
        
        global_emb = self.global_fusion(global_emb)
        global_emb = global_emb.unsqueeze(1).expand(-1, self.max_atoms, -1)
        
        # 5. Cross-scale interaction (only when both use_coarse and cross_scale_interaction are True)
        if self.cross_scale_interaction and self.use_coarse:
            atom_to_coarse = multiscale_features['atom_to_coarse_mapping'].to(self.device)
            
            fine_emb, coarse_emb = self.cross_scale_fusion(
                fine_emb, 
                coarse_emb, 
                atom_to_coarse,
                global_emb
            )
        
        # 6. Pooling to obtain molecular representation
        fine_mask = (torch.sum(multiscale_features['fine_x'], dim=-1) != 0).to(self.device).float()
        fine_mask = fine_mask.unsqueeze(-1)
        fine_pooled = torch.sum(fine_emb * fine_mask, dim=1) / (torch.sum(fine_mask, dim=1) + 1e-8)
        
        if self.use_coarse:  # Compute coarse pooling only when coarse is used, otherwise set to 0
            coarse_mask = (torch.sum(multiscale_features['coarse_node_features'], dim=-1) != 0).to(self.device).float()
            coarse_mask = coarse_mask.unsqueeze(-1)
            coarse_pooled = torch.sum(coarse_emb * coarse_mask, dim=1) / (torch.sum(coarse_mask, dim=1) + 1e-8)
        else:
            coarse_pooled = 0  # Contribution from coarse-grained part is 0 when disabled

        # 7. Fuse multi-scale representations (including sequence)
        combined = fine_pooled + coarse_pooled + 0.5 * seq_emb
        combined = self.output_proj(combined)
        
        return combined.unsqueeze(1).expand(-1, self.max_atoms, -1)

class UniDTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(UniDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']
        
        # Multi-grain parameters
        self.use_multigrain = config["DRUG"]["MULTIGRAIN"]["ENABLE"]
        max_coarse_nodes = config["DRUG"]["MULTIGRAIN"]["MAX_COARSE_NODES"]
        coarse_node_dim = config["DRUG"]["MULTIGRAIN"]["COA_RSE_NODE_DIM"]
        atom_dim = config["DRUG"]["MULTIGRAIN"]["ATOM_DIM"]
        ecfp_dim = config["DRUG"]["MULTIGRAIN"]["ECFP_DIM"]
        desc_dim = config["DRUG"]["MULTIGRAIN"]["DESC_DIM"]
        self.cross_scale_interaction = config["DRUG"]["MULTIGRAIN"]["CROSS_SCALE_INTERACTION"]
        self.use_global_prop = config["DRUG"]["MULTIGRAIN"]["USE_GLOBAL_PROP"]
        
        # Sequence parameters
        self.use_drug_sequence = config["DRUG"]["SEQUENCE"]["ENABLE"]
        smiles_vocab_size = config["DRUG"]["SEQUENCE"]["SMILES_VOCAB_SIZE"]
        iupac_vocab_size = config["DRUG"]["SEQUENCE"]["IUPAC_VOCAB_SIZE"]
        
        # Protein GNN parameters
        self.use_protein_gnn = config["PROTEIN"]["GNN"]["ENABLE"]
        gnn_hidden_dim = config["PROTEIN"]["GNN"]["HIDDEN_DIM"]
        gnn_layers = config["PROTEIN"]["GNN"]["NUM_LAYERS"]
        gnn_dropout = config["PROTEIN"]["GNN"]["DROPOUT"]

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        
        # If multi-grain is enabled, use the enhanced multi-grain encoder
        if self.use_multigrain:
            self.multigrain_drug_extractor = MultigrainMolecularEncoder(
                atom_dim=atom_dim,
                coarse_node_dim=coarse_node_dim,
                ecfp_dim=ecfp_dim,
                desc_dim=desc_dim,
                hidden_dim=drug_embedding,
                max_atoms=config["DRUG"]["MAX_NODES"],
                max_coarse_nodes=max_coarse_nodes,
                cross_scale_interaction=self.cross_scale_interaction,
                use_global_prop=self.use_global_prop,
                device=device,
                smiles_vocab_size=smiles_vocab_size,
                iupac_vocab_size=iupac_vocab_size,
                use_coarse=config["DRUG"]["MULTIGRAIN"]["USE_COARSE"]  # New parameter to control use of coarse-grained features
            )
            self.drug_fusion = nn.Sequential(
                nn.Linear(drug_embedding * 2, drug_embedding),
                nn.LayerNorm(drug_embedding),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        
        # Protein encoder (ACmix + GCN)
        if self.use_protein_gnn:
            self.protein_extractor = EnhancedProteinEncoder(
                protein_emb_dim, 
                num_filters, 
                protein_num_head, 
                protein_padding,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_layers=gnn_layers,
                gnn_dropout=gnn_dropout
            )
        else:
            self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, 
                                          layer=cross_layer, device=device)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, multiscale_features=None, drug_seq=None, iupac_seq=None, 
                protein_graph=None, mode="train"):

        v_d = self.drug_extractor(bg_d)
        
        # If multi-grain is enabled, extract multi-grain features (including sequences)
        if self.use_multigrain and multiscale_features is not None:
            if self.use_drug_sequence and drug_seq is not None and iupac_seq is not None:
                multigrain_v_d = self.multigrain_drug_extractor(multiscale_features, drug_seq, iupac_seq)
            else:
                multigrain_v_d = self.multigrain_drug_extractor(multiscale_features)
            
            # Fuse fine-grained and multi-grain features
            v_d = torch.cat([v_d, multigrain_v_d], dim=-1)
            v_d = self.drug_fusion(v_d)
        
        # Protein encoding (including GNN)
        if self.use_protein_gnn and protein_graph is not None:
            v_p = self.protein_extractor(v_p, protein_graph)
        else:
            v_p = self.protein_extractor(v_p)

        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)
        score = self.mlp_classifier(f)
        
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

# The following classes remain unchanged (MolecularGCN, ProteinACmix, CrossScaleFusion, MLPDecoder)
class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class ProteinACmix(nn.Module):
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(ProteinACmix, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]

        self.acmix1 = ACmix(in_planes=in_ch[0], out_planes=in_ch[1], head=num_head)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.acmix2 = ACmix(in_planes=in_ch[1], out_planes=in_ch[2], head=num_head)
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.acmix3 = ACmix(in_planes=in_ch[2], out_planes=in_ch[3], head=num_head)
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = torch.clamp(v.long(), 0, 25)  # Clamp amino acid indices (0-25)
        v = self.embedding(v)
        v = v.transpose(2, 1)

        v = self.bn1(F.relu(self.acmix1(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn2(F.relu(self.acmix2(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn3(F.relu(self.acmix3(v.unsqueeze(-2))).squeeze(-2))

        v = v.view(v.size(0), v.size(2), -1)
        return v

class CrossScaleFusion(nn.Module):
    """Cross-scale interaction and fusion module"""
    
    def __init__(self, hidden_dim=128, dropout=0.3, device='cuda'):
        super(CrossScaleFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Pooling layer (fine-grained -> coarse-grained)
        self.fine_to_coarse_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Unpooling layer (coarse-grained -> fine-grained)
        self.coarse_to_fine_unpool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Global information injection
        self.global_injection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, fine_features, coarse_features, atom_to_coarse, global_features):
        batch_size, num_atoms, _ = fine_features.shape
        _, num_coarse_nodes, _ = coarse_features.shape
        
        # 1. Fine-grained -> coarse-grained (Pooling)
        coarse_from_fine = torch.zeros(batch_size, num_coarse_nodes, self.hidden_dim, device=self.device)
        coarse_counts = torch.zeros(batch_size, num_coarse_nodes, device=self.device)
        
        for b in range(batch_size):
            valid_mask = (atom_to_coarse[b] >= 0) & (atom_to_coarse[b] < num_coarse_nodes)
            valid_indices = atom_to_coarse[b][valid_mask]
            valid_fine_features = fine_features[b, valid_mask]
            
            if len(valid_indices) > 0:
                expanded_indices = valid_indices.view(-1, 1).expand(-1, self.hidden_dim)
                coarse_from_fine[b].scatter_add_(0, expanded_indices, valid_fine_features)
                coarse_counts[b].scatter_add_(0, valid_indices, torch.ones_like(valid_indices, dtype=torch.float))
        
        coarse_counts = torch.clamp(coarse_counts, min=1.0)
        coarse_from_fine = coarse_from_fine / coarse_counts.unsqueeze(-1)
        coarse_from_fine = self.fine_to_coarse_pool(coarse_from_fine)
        
        # 2. Coarse-grained -> fine-grained (Unpooling)
        fine_from_coarse = torch.zeros_like(fine_features)
        
        for b in range(batch_size):
            valid_mask = (atom_to_coarse[b] >= 0) & (atom_to_coarse[b] < num_coarse_nodes)
            valid_indices = atom_to_coarse[b][valid_mask]
            
            if len(valid_indices) > 0:
                fine_from_coarse[b, valid_mask] = coarse_features[b, valid_indices]
        
        fine_from_coarse = self.coarse_to_fine_unpool(fine_from_coarse)
        
        # 3. Gated fusion
        fine_gate = self.gate(torch.cat([fine_features, fine_from_coarse], dim=-1))
        fine_updated = fine_gate * fine_features + (1 - fine_gate) * fine_from_coarse
        
        coarse_gate = self.gate(torch.cat([coarse_features, coarse_from_fine], dim=-1))
        coarse_updated = coarse_gate * coarse_features + (1 - coarse_gate) * coarse_from_fine
        
        # 4. Global information injection
        if global_features is not None:
            fine_with_global = self.global_injection(torch.cat([fine_updated, global_features], dim=-1))
            fine_updated = fine_updated + 0.1 * fine_with_global
            
            global_coarse = torch.mean(global_features, dim=1, keepdim=True).expand(-1, num_coarse_nodes, -1)
            coarse_with_global = self.global_injection(torch.cat([coarse_updated, global_coarse], dim=-1))
            coarse_updated = coarse_updated + 0.1 * coarse_with_global
        
        return fine_updated, coarse_updated

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x