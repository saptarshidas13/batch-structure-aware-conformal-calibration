import anndata as ad

a = ad.read_h5ad(r"D:\Research\NCS\phase1_baseline\data\processed\baron_human1.h5ad")
print(a)
print("X max/min:", a.X.max(), a.X.min())
print("obs columns:", a.obs.columns.tolist())
print(a.obs.head())
