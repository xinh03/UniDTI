# multigrain.py - 多粒度分子特征提取模块
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings
import networkx as nx

warnings.filterwarnings("ignore")

class MultigrainMolecularFeatures:
    """多粒度分子特征提取器"""
    
    # 原子类型定义
    ATOM_TYPES = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'OTHER']
    ATOM_TYPE_TO_IDX = {t: i for i, t in enumerate(ATOM_TYPES)}
    
    # 粗粒度节点类型
    COARSE_NODE_TYPES = ['RING', 'FUNCTIONAL_GROUP', 'NON_CORE']
    COARSE_NODE_TO_IDX = {t: i for i, t in enumerate(COARSE_NODE_TYPES)}
    
    @staticmethod
    def get_atom_features(atom):
        """提取原子特征（34维）"""
        try:
            atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
            atom_symbol = atom.GetSymbol()
            atom_type_onehot = [int(atom_symbol == t) for t in atom_types]
            if sum(atom_type_onehot) == 0:
                atom_type_onehot.append(1)  # 其他原子类型
            else:
                atom_type_onehot.append(0)
            
            features = atom_type_onehot
            
            # 原子度
            degree = [0] * 6
            degree[min(atom.GetDegree(), 5)] = 1
            features.extend(degree)
            
            features.append(atom.GetFormalCharge())  # 形式电荷
            
            # 手性
            chiral = [0] * 4
            chiral_tag = atom.GetChiralTag()
            chiral_idx = min(int(chiral_tag), 3)
            chiral[chiral_idx] = 1
            features.extend(chiral)
            
            # 氢原子数
            num_h = [0] * 5
            num_h[min(atom.GetTotalNumHs(), 4)] = 1
            features.extend(num_h)
            
            # 杂化类型
            hybridization_types = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2, 
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ]
            hybridization = [int(atom.GetHybridization() == h) for h in hybridization_types]
            features.extend(hybridization)
            
            features.append(int(atom.GetIsAromatic()))  # 是否芳香族
            features.append(atom.GetMass() / 100.0)     # 原子质量(归一化)
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(34, dtype=np.float32)
    
    @staticmethod
    def get_coarse_node_features(mol, atom_indices, node_type):
        """提取粗粒度节点特征"""
        try:
            features = []
            
            # 节点类型one-hot
            type_onehot = [0] * len(MultigrainMolecularFeatures.COARSE_NODE_TO_IDX)
            if node_type in MultigrainMolecularFeatures.COARSE_NODE_TO_IDX:
                type_onehot[MultigrainMolecularFeatures.COARSE_NODE_TO_IDX[node_type]] = 1
            else:
                type_onehot[0] = 1  # 默认为RING类型
            
            features.extend(type_onehot)
            
            # 原子组成统计（10种原子类型）
            atom_counts = []
            for atom_type in ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'OTHER']:
                count = 0
                for atom_idx in atom_indices:
                    if atom_idx < mol.GetNumAtoms():
                        atom = mol.GetAtomWithIdx(atom_idx)
                        if atom.GetSymbol() == atom_type:
                            count += 1
                        elif atom_type == 'OTHER' and atom.GetSymbol() not in ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']:
                            count += 1
                atom_counts.append(count)
            features.extend(atom_counts)
            
            # 节点大小
            features.append(len(atom_indices))
            
            # 是否包含芳香原子
            aromatic = 0
            for atom_idx in atom_indices:
                if atom_idx < mol.GetNumAtoms():
                    if mol.GetAtomWithIdx(atom_idx).GetIsAromatic():
                        aromatic = 1
                        break
            features.append(aromatic)
            
            return np.array(features, dtype=np.float32)
        except:
            default_features = np.zeros(len(MultigrainMolecularFeatures.COARSE_NODE_TO_IDX) + 10 + 2, dtype=np.float32)
            default_features[0] = 1.0  # 默认为RING类型
            return default_features
    
    @staticmethod
    def get_ring_systems(mol, include_spiro=False):
        """提取环系统"""
        try:
            ri = mol.GetRingInfo()
            systems = []
            for ring in ri.AtomRings():
                ringAts = set(ring)
                nSystems = []
                for system in systems:
                    nInCommon = len(ringAts.intersection(system))
                    if nInCommon and (include_spiro or nInCommon > 1):
                        ringAts = ringAts.union(system)
                    else:
                        nSystems.append(system)
                nSystems.append(ringAts)
                systems = nSystems
            return list(systems)
        except:
            return []
    
    @staticmethod
    def get_ertl_functional_groups(mol):
        """提取Ertl官能团"""
        try:
            marked_atoms = set()
            
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                atomic_num = atom.GetAtomicNum()
                
                if idx in marked_atoms:
                    continue

                # 处理芳香性杂原子（排除芳香碳）
                if atom.GetIsAromatic():
                    if atomic_num != 6:
                        marked_atoms.add(idx)
                    continue

                # 处理碳原子
                if atomic_num == 6:
                    is_marked = False
                    neighbors = atom.GetNeighbors()
                    single_bond_hetero_count = 0
                    
                    for neighbor in neighbors:
                        n_idx = neighbor.GetIdx()
                        bond = mol.GetBondBetweenAtoms(idx, n_idx)
                        bond_order = bond.GetBondType()
                        n_atomic_num = neighbor.GetAtomicNum()
                        
                        # C=X, C#X 不饱和键
                        if (n_atomic_num != 1 and 
                            (bond_order == Chem.BondType.DOUBLE or bond_order == Chem.BondType.TRIPLE) and
                            not bond.GetIsAromatic()):
                            marked_atoms.add(n_idx)
                            is_marked = True
                        
                        # 单键杂原子计数
                        elif n_atomic_num in [7, 8, 16] and bond_order == Chem.BondType.SINGLE:
                            if not neighbor.GetIsAromatic():
                                marked_atoms.add(n_idx)
                                # 检查杂原子是否只连单键
                                is_all_single = True
                                for n_neighbor in neighbor.GetNeighbors():
                                    if mol.GetBondBetweenAtoms(n_idx, n_neighbor.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
                                        is_all_single = False
                                if is_all_single:
                                    single_bond_hetero_count += 1

                    if single_bond_hetero_count > 1:
                        is_marked = True
                        
                    if atom.IsInRingSize(3):  # 小环
                        for neighbor in neighbors:
                            if neighbor.GetAtomicNum() not in [1, 6] and neighbor.IsInRingSize(3):
                                is_marked = True
                                marked_atoms.add(neighbor.GetIdx())

                    if is_marked:
                        marked_atoms.add(idx)

                # 其他杂原子
                elif atomic_num != 1:
                    marked_atoms.add(idx)

            # 提取连通分量
            if not marked_atoms:
                return []
            
            G = nx.Graph()
            for idx in marked_atoms:
                G.add_node(idx)
            
            for bond in mol.GetBonds():
                b_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if b_idx in marked_atoms and e_idx in marked_atoms:
                    G.add_edge(b_idx, e_idx)
                    
            return [list(c) for c in nx.connected_components(G)]
        except:
            return []
    
    @staticmethod
    def get_non_core_structures(mol, ring_indices_list, fg_indices_list):
        """提取Non-Core结构"""
        try:
            all_atoms = set(range(mol.GetNumAtoms()))
            
            # 收集所有被占用的原子
            occupied_atoms = set()
            for r in ring_indices_list:
                occupied_atoms.update(r)
            for fg in fg_indices_list:
                occupied_atoms.update(fg)
            
            # 计算差集
            non_core_atoms = all_atoms - occupied_atoms
            
            if not non_core_atoms:
                return []

            # 将离散的原子聚类成连通结构
            G = nx.Graph()
            for idx in non_core_atoms:
                G.add_node(idx)
            
            for bond in mol.GetBonds():
                b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if b in non_core_atoms and e in non_core_atoms:
                    G.add_edge(b, e)
                    
            return [list(c) for c in nx.connected_components(G)]
        except:
            return []
    
    @staticmethod
    def extract_multiscale_graph_features(smiles, max_atoms=150, max_coarse_nodes=50):
        """提取多尺度图特征（细粒度和粗粒度）"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 1. 细粒度图特征（原子级）
        num_atoms = min(mol.GetNumAtoms(), max_atoms)
        
        # 原子特征
        fine_x = []
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            features = MultigrainMolecularFeatures.get_atom_features(atom)
            fine_x.append(features)
        
        # 填充或截断
        if len(fine_x) < max_atoms:
            pad_len = max_atoms - len(fine_x)
            fine_x.extend([np.zeros(34, dtype=np.float32)] * pad_len)
        else:
            fine_x = fine_x[:max_atoms]
        fine_x = np.array(fine_x, dtype=np.float32)
        
        # 细粒度图边特征
        fine_edges = []
        fine_edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            if i >= max_atoms or j >= max_atoms:
                continue
            
            fine_edges.append([i, j])
            fine_edges.append([j, i])
            
            bond_type = bond.GetBondType()
            if bond_type == Chem.rdchem.BondType.SINGLE:
                attr = [1.0]
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                attr = [2.0]
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                attr = [3.0]
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                attr = [1.5]
            else:
                attr = [1.0]
            
            fine_edge_attrs.append(attr)
            fine_edge_attrs.append(attr)
        
        if fine_edges:
            fine_edge_index = np.array(fine_edges, dtype=np.int64).T
            fine_edge_attr = np.array(fine_edge_attrs, dtype=np.float32)
        else:
            fine_edge_index = np.zeros((2, 0), dtype=np.int64)
            fine_edge_attr = np.zeros((0, 1), dtype=np.float32)
        
        # 2. 粗粒度结构提取
        ring_indices = MultigrainMolecularFeatures.get_ring_systems(mol)
        fg_indices = MultigrainMolecularFeatures.get_ertl_functional_groups(mol)
        non_core_indices = MultigrainMolecularFeatures.get_non_core_structures(mol, ring_indices, fg_indices)
        
        # 收集所有粗粒度节点
        coarse_atom_indices = []
        coarse_node_types = []
        
        for ring in ring_indices:
            if len(ring) > 0:
                coarse_atom_indices.append(list(ring))
                coarse_node_types.append('RING')
        
        for fg in fg_indices:
            if len(fg) > 0:
                coarse_atom_indices.append(list(fg))
                coarse_node_types.append('FUNCTIONAL_GROUP')
        
        for nc in non_core_indices:
            if len(nc) > 0:
                coarse_atom_indices.append(list(nc))
                coarse_node_types.append('NON_CORE')
        
        # 限制最大粗粒度节点数
        if len(coarse_atom_indices) > max_coarse_nodes:
            coarse_atom_indices = coarse_atom_indices[:max_coarse_nodes]
            coarse_node_types = coarse_node_types[:max_coarse_nodes]
        
        # 3. 原子到粗粒度节点映射
        atom_to_coarse = np.full(max_atoms, -1, dtype=np.int64)
        for coarse_idx, atom_indices in enumerate(coarse_atom_indices):
            for atom_idx in atom_indices:
                if atom_idx < max_atoms:
                    atom_to_coarse[atom_idx] = coarse_idx
        
        # 4. 构建粗粒度图边
        coarse_edges = []
        coarse_edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            if i >= max_atoms or j >= max_atoms:
                continue
            
            coarse_i = atom_to_coarse[i]
            coarse_j = atom_to_coarse[j]
            
            if coarse_i != -1 and coarse_j != -1 and coarse_i != coarse_j:
                edge_exists = False
                for k in range(len(coarse_edges)):
                    if (coarse_edges[k][0] == coarse_i and coarse_edges[k][1] == coarse_j) or \
                       (coarse_edges[k][0] == coarse_j and coarse_edges[k][1] == coarse_i):
                        edge_exists = True
                        break
                
                if not edge_exists:
                    coarse_edges.append([coarse_i, coarse_j])
                    coarse_edges.append([coarse_j, coarse_i])
                    
                    bond_type = bond.GetBondType()
                    if bond_type == Chem.rdchem.BondType.SINGLE:
                        attr = [1.0]
                    elif bond_type == Chem.rdchem.BondType.DOUBLE:
                        attr = [2.0]
                    elif bond_type == Chem.rdchem.BondType.TRIPLE:
                        attr = [3.0]
                    else:
                        attr = [1.0]
                    
                    coarse_edge_attrs.append(attr)
                    coarse_edge_attrs.append(attr)
        
        if coarse_edges:
            coarse_edge_index = np.array(coarse_edges, dtype=np.int64).T
            coarse_edge_attr = np.array(coarse_edge_attrs, dtype=np.float32)
        else:
            coarse_edge_index = np.zeros((2, 0), dtype=np.int64)
            coarse_edge_attr = np.zeros((0, 1), dtype=np.float32)
        
        # 5. 粗粒度节点特征
        coarse_node_features = []
        for atom_indices, node_type in zip(coarse_atom_indices, coarse_node_types):
            features = MultigrainMolecularFeatures.get_coarse_node_features(mol, atom_indices, node_type)
            coarse_node_features.append(features)
        
        # 填充粗粒度节点特征
        if len(coarse_node_features) < max_coarse_nodes:
            pad_len = max_coarse_nodes - len(coarse_node_features)
            pad_features = np.zeros((pad_len, len(MultigrainMolecularFeatures.COARSE_NODE_TO_IDX) + 10 + 2), dtype=np.float32)
            coarse_node_features.extend(pad_features)
        coarse_node_features = np.array(coarse_node_features, dtype=np.float32)
        
        # 6. 分子指纹和描述符
        try:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            ecfp_array = np.array(ecfp, dtype=np.float32)
        except:
            ecfp_array = np.zeros(1024, dtype=np.float32)
        
        try:
            descriptors = np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
            ], dtype=np.float32)
        except:
            descriptors = np.zeros(8, dtype=np.float32)
        
        # 填充描述符到200维
        if len(descriptors) < 200:
            descriptors = np.pad(descriptors, (0, 200 - len(descriptors)), 'constant')
        
        return {
            'fine_x': fine_x,
            'fine_edge_index': fine_edge_index,
            'fine_edge_attr': fine_edge_attr,
            'coarse_node_features': coarse_node_features,
            'coarse_edge_index': coarse_edge_index,
            'coarse_edge_attr': coarse_edge_attr,
            'atom_to_coarse_mapping': atom_to_coarse,
            'ecfp': ecfp_array,
            'descriptors': descriptors,
            'num_atoms': num_atoms,
            'num_coarse_nodes': len(coarse_atom_indices)
        }