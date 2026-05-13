from src.model.dataset import SSHDiffusionDataset
ds = SSHDiffusionDataset("train")
print(f"cond_dim={ds.cond_dim}  target_dim={ds.target_dim}")
cond, target = ds[0]
print(f"cond.shape={tuple(cond.shape)}  target.shape={tuple(target.shape)}")
