import pandas as pd

path = "outputs/LFPO/tiles/Lfpo_tile_1_2.parquet"  # adapte au vrai chemin

df = pd.read_parquet(path)

print("Shape:", df.shape)
print("Min:")
print(df[["x", "y", "z", "s"]].min())
print("Max:")
print(df[["x", "y", "z", "s"]].max())