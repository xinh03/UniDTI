# generate_prot_contact_map.py
# Author  : Ethan Huang
# File    : generate_prot_contact_map.py
# Time    : 2026/3/25

"""
env: conda activate unidti

python generate_prot_contact_map.py \
  --pdb_dir /path/to/your/pdbs \
  --p2rank_dir /path/to/your/p2rank_results \
  --output_dir /path/to/your/output_contact_maps \
  --top_k 3 # Optional, default to 3, means keep the first K pockets

"""

import os
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate protein contact map from P2Rank predictions.csv (with full-protein fallback)")
    
    parser.add_argument('--pdb_dir', type=str, required=True, 
                        help='Path to directory containing .pdb files')
    parser.add_argument('--p2rank_dir', type=str, required=True, 
                        help='Path to directory containing P2Rank results (*_predictions.csv)')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Path to save generated .npy contact maps')
    parser.add_argument('--threshold', type=float, default=8.0, 
                        help='Distance threshold for contact map (Angstrom), default 8.0')
    parser.add_argument('--top_k', type=int, default=3, 
                        help='Number of top-ranked pockets to keep, default 3')
    
    return parser.parse_args()

def setup_logging(output_dir):
    """Configure logging system: write information and warnings to a log file."""
    log_file = os.path.join(output_dir, 'processing_log.log')
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    print(f"Detailed log will be saved to: {log_file}")

def get_pocket_residue_ids_from_predictions(csv_path, pdb_id, top_k=3):
    """
    Parse P2Rank predictions.csv to retrieve pocket residue ids.
    
    Returns:
        None: file read error or severe format error
        set(): file is valid but no pocket matching the rank criteria (empty set)
        set({...}): pocket residues successfully extracted
    """
    if not os.path.exists(csv_path):
        logging.warning(f"[{pdb_id}] CSV file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        if 'rank' not in df.columns or 'residue_ids' not in df.columns:
            logging.warning(f"[{pdb_id}] CSV missing 'rank' or 'residue_ids' column")
            return set()

        target_df = df[df['rank'] <= top_k]
        
        if target_df.empty:
            return set()

        all_residues = set()
        
        for _, row in target_df.iterrows():
            ids_str = str(row['residue_ids']).strip()
            
            if not ids_str or ids_str.lower() == 'nan':
                continue
                
            tokens = ids_str.split()
            for token in tokens:
                try:
                    if '_' in token:
                        res_num_str = token.split('_')[-1]
                    else:
                        res_num_str = token
                    
                    res_num = int(res_num_str)
                    all_residues.add(res_num)
                except ValueError:
                    continue

        return all_residues
        
    except Exception as e:
        logging.error(f"[{pdb_id}] Exception while reading CSV: {e}")
        return None

def calc_contact_map(structure, target_residue_ids, pdb_id, threshold=8.0):
    """
    Read PDB structure and compute contact map.
    If target_residue_ids is None, compute for the whole protein.
    """
    coords = []
    
    try:
        model = structure[0]
    except Exception as e:
        logging.error(f"[{pdb_id}] PDB structure is empty or model cannot be obtained: {e}")
        return None
    
    # Iterate over all chains and residues
    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            
            res_id = residue.id[1]
            
            # Core logic: if target_residue_ids is None, use the full protein (no filtering);
            # otherwise keep only residues present in the pocket set.
            if target_residue_ids is not None:
                if res_id not in target_residue_ids:
                    continue
            
            try:
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
                elif 'CB' in residue:
                    coords.append(residue['CB'].get_coord())
                else:
                    atom_coords = [atom.get_coord() for atom in residue]
                    center = np.mean(atom_coords, axis=0)
                    coords.append(center)
            except Exception:
                continue

    if not coords:
        logging.warning(f"[{pdb_id}] No valid atomic coordinates extracted")
        return None

    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    contact_map = (dist_matrix < threshold).astype(np.int8)
    
    return contact_map

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    parser = PDBParser(QUIET=True)
    
    pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} PDB files, starting processing...")
    
    success_count = 0
    fallback_count = 0  # number of proteins processed in full-protein mode
    skip_count = 0
    
    for pdb_file in tqdm(pdb_files):
        pdb_path = os.path.join(args.pdb_dir, pdb_file)
        pdb_id = pdb_file.replace('.pdb', '')
        
        p2rank_csv_name = f"{pdb_file}_predictions.csv"
        p2rank_csv_path = os.path.join(args.p2rank_dir, p2rank_csv_name)
        
        # 1. Try to locate the P2Rank result file
        csv_exists = True
        if not os.path.exists(p2rank_csv_path):
            alt_path = os.path.join(args.p2rank_dir, f"{pdb_id}_predictions.csv")
            if os.path.exists(alt_path):
                p2rank_csv_path = alt_path
            else:
                logging.warning(f"[{pdb_id}] Prediction file not found, using full-protein mode")
                csv_exists = False
                
        target_res_ids = None  # defaults to None (full-protein mode)
        
        # 2. If the file exists, try to parse pocket residues
        if csv_exists:
            found_ids = get_pocket_residue_ids_from_predictions(p2rank_csv_path, pdb_id, top_k=args.top_k)
            
            if found_ids is None:
                # return None means file read error -> skip this protein
                skip_count += 1
                continue
            elif len(found_ids) == 0:
                # empty set means no suitable pocket -> fallback to full-protein mode
                logging.info(f"[{pdb_id}] No pocket found with rank <= {args.top_k}, switching to full-protein mode")
                target_res_ids = None
                fallback_count += 1
            else:
                # pocket successfully found
                target_res_ids = found_ids
        else:
            # file does not exist, also counts as fallback
            fallback_count += 1
            
        # 3. Compute the contact map (if target_res_ids is None, the function automatically uses the full protein)
        try:
            structure = parser.get_structure(pdb_id, pdb_path)
            cmap = calc_contact_map(structure, target_res_ids, pdb_id, threshold=args.threshold)
            
            if cmap is not None:
                save_path = os.path.join(args.output_dir, f"{pdb_id}.npy")
                np.save(save_path, cmap)
                success_count += 1
            else:
                skip_count += 1
                
        except Exception as e:
            logging.error(f"[{pdb_id}] Processing exception: {e}")
            skip_count += 1

    print("\n" + "="*30)
    print(f"Processing Summary:")
    print(f"Total successfully generated: {success_count}")
    print(f"  - Pocket-based: {success_count - fallback_count}")
    print(f"  - Whole-protein (fallback): {fallback_count}")
    print(f"Skipped/Failed: {skip_count}")
    print(f"Log saved to: {os.path.join(args.output_dir, 'processing_log.log')}")
    print("="*30)

if __name__ == '__main__':
    main()