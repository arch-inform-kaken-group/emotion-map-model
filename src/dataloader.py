import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import ASSIGNED_NUMBERS_DICT


# Finds all valid 'qa_corrected.csv' paths based on the predefined pottery IDs.
def find_data_paths_detailed(root, pottery_path_str, limit=1000):
    root, pottery_path = Path(root), Path(pottery_path_str)
    if not root.exists():
        raise ValueError(f"Root directory not found: {root}")
    if not pottery_path.exists():
        raise ValueError(f"Pottery directory not found: {pottery_path}")
    
    data = []
    pottery_ids = [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]
    
    print(f"\nCHECKING RAW DATA PATHS")
    limit_dict = {pid: 0 for pid in pottery_ids}
    
    for g in os.listdir(root):
        group_path = root / g
        if not os.path.isdir(group_path): continue
        for s in tqdm(os.listdir(group_path), desc=g):
            session_path = group_path / s
            if not os.path.isdir(session_path): continue
            for p in os.listdir(session_path):
                if p in pottery_ids and limit_dict[p] < limit:
                    qa_path = session_path / p / "qa_corrected.csv"
                    if qa_path.exists():
                        limit_dict[p] += 1
                        data.append({
                            'qa': str(qa_path),
                            'GROUP': g,
                            'SESSION_ID': s,
                            'ID': p
                        })

    print(f"\nLoader finished. Found {len(data)} valid data instances.")
    return data


# Loads and combines all found Emotion record data into a single DataFrame.
def load_combined_qna_data(root_dir, pottery_models_dir):
    data_to_process = find_data_paths_detailed(root=root_dir, pottery_path_str=pottery_models_dir)
    if not data_to_process: 
        return pd.DataFrame()

    df_list = []
    for item in tqdm(data_to_process, desc="Loading and combining data"):
        try:
            temp_df = pd.read_csv(item['qa'], header=0, sep=",")
            temp_df['timestamp'] = pd.to_numeric(temp_df['timestamp'], errors='coerce')
            temp_df.dropna(subset=['timestamp'], inplace=True)
            temp_df['pottery_id'] = item['ID']
            temp_df['session_id'] = item['SESSION_ID']
            df_list.append(temp_df)
        except Exception as e:
            print(f"Could not read or process file {item['qa']}: {e}", file=sys.stderr)

    if not df_list: 
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)