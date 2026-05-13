"""
Plot mean SST and mean SSH for January 1995 over the US East Coast.
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_FILE = os.path.join(DATA_DIR, "mean_sst_ssh_199501.png")

sst = xr.open_dataset(os.path.join(DATA_DIR, "glorys_sst_199501_east_coast.nc"))
ssh = xr.open_dataset(os.path.join(DATA_DIR, "glorys_ssh_199501_east_coast.nc"))

mean_sst = sst["thetao"].mean("time").squeeze()
mean_ssh = ssh["zos"].mean("time").squeeze()

proj = ccrs.PlateCarree()
land = cfeature.NaturalEarthFeature("physical", "land", "50m",
                                    facecolor="lightgray", edgecolor="black", linewidth=0.5)

fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         subplot_kw={"projection": proj})

extent = [-85, -50, 25, 50]

def add_map_features(ax):
    ax.set_extent(extent, crs=proj)
    ax.add_feature(land)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(10)
    gl.ylocator = mticker.MultipleLocator(5)

# ── SST ──────────────────────────────────────────────────────────────────────
ax = axes[0]
add_map_features(ax)
p = ax.pcolormesh(mean_sst.longitude, mean_sst.latitude, mean_sst,
                  cmap=cmocean.cm.thermal, transform=proj, shading="auto")
cb = fig.colorbar(p, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85)
cb.set_label("SST (°C)")
ax.set_title("Mean Sea Surface Temperature\nJanuary 1995", fontsize=11)

# ── SSH ──────────────────────────────────────────────────────────────────────
ax = axes[1]
add_map_features(ax)
p = ax.pcolormesh(mean_ssh.longitude, mean_ssh.latitude, mean_ssh,
                  cmap=cmocean.cm.balance, transform=proj, shading="auto",
                  norm=plt.Normalize(vmin=float(mean_ssh.min()),
                                     vmax=float(mean_ssh.max())))
cb = fig.colorbar(p, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85)
cb.set_label("SSH (m)")
ax.set_title("Mean Sea Surface Height\nJanuary 1995", fontsize=11)

fig.suptitle("GLORYS12 — US East Coast", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
plt.savefig(OUT_FILE.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
