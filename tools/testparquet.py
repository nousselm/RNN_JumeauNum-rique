# import pandas as pd


# df = pd.read_parquet("data/LFPO_sdf_train.parquet")
# print(df[["x", "y", "z"]].min())
# print(df[["x", "y", "z"]].max())

# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

import pandas as pd

def count_points(file_path):
    df = pd.read_parquet(file_path)
    
    print("Colonnes :", df.columns.tolist())
    print("Shape :", df.shape)
    print("Nombre de points :", len(df))
    
    return len(df)

