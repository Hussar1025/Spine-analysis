import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = r"yourpath"

COUNTS_CSV = os.path.join(OUT_DIR, "yourdata")
CELL_META_CSV = os.path.join(OUT_DIR, "yourdata")
GENE_META_CSV = os.path.join(OUT_DIR, "yourdata")
DYNAMIC_GENES_CSV = os.path.join(OUT_DIR, "yourdata")
PSEUDOTIME_CSV = os.path.join(OUT_DIR, "yourdata")
H5AD_PATH = os.path.join(OUT_DIR, "yourdata")

MODEL_PATH = os.path.join(OUT_DIR, "yourdata")
SAVE_Y_DYNAMIC = os.path.join(OUT_DIR, "yourdata")
SAVE_PT_DYNAMIC = os.path.join(OUT_DIR, "yourdata")
SAVE_COORDS = os.path.join(OUT_DIR, "yourdata")
SAVE_CELL_IDS = os.path.join(OUT_DIR, "yourdata")
SAVE_GENES = os.path.join(OUT_DIR, "yourdata")

N_CELLS = 20000
N_GENES = 200
K_SPACE = 20
N_STEPS = 8
EPOCHS = 25
LR = 2e-3
WEIGHT_DECAY = 1e-5
SEED = 42
HIDDEN_DIM = 64
HEADS = 2
TEACHER_FORCING_RATIO = 0.1
STEP_SIZE = 0.1
NOISE_STD = 0.05
DIVERSITY_LAMBDA = 0.4
CLAMP_MAX = 3.0
MEAN_REG_LAMBDA = 0.1
PT_LOSS_WEIGHT = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

adata = sc.read_h5ad(H5AD_PATH)
if "spatial" in adata.obsm:
    all_coords = adata.obsm["spatial"]
else:
    x_col = next(c for c in ["x", "X", "x_centroid", "center_x", "spatial_x"] if c in adata.obs.columns)
    y_col = next(c for c in ["y", "Y", "y_centroid", "center_y", "spatial_y"] if c in adata.obs.columns)
    all_coords = adata.obs[[x_col, y_col]].values

adata_cell_ids = adata.obs_names.astype(str).str.strip().to_numpy()
adata_index_map = {c: i for i, c in enumerate(adata_cell_ids)}

cell_meta = pd.read_csv(CELL_META_CSV, index_col=0)
gene_meta = pd.read_csv(GENE_META_CSV, index_col=0)
pt_df = pd.read_csv(PSEUDOTIME_CSV)
counts_df = pd.read_csv(COUNTS_CSV, index_col=0)

gene_names_meta = gene_meta.index.astype(str).str.strip()
if len(set(counts_df.index) & set(gene_names_meta)) > len(set(counts_df.columns) & set(gene_names_meta)):
    counts_df = counts_df.T

counts_df.index = counts_df.index.astype(str).str.strip()
counts_df.columns = counts_df.columns.astype(str).str.strip()

cell_col = next(c for c in pt_df.columns if "cell" in c.lower() or "barcode" in c.lower())
pt_col = next(c for c in pt_df.columns if "pseudo" in c.lower() or c.lower() in ["pt", "pseudotime"])
pt_df[cell_col] = pt_df[cell_col].astype(str).str.strip()
pt_map = dict(zip(pt_df[cell_col], pt_df[pt_col].astype(float)))

common_cells = list(set(counts_df.index) & set(pt_map.keys()) & set(adata_index_map.keys()))
if len(common_cells) == 0:
    raise ValueError("No common cells found. Check ID formatting.")

pt_all = np.array([pt_map[c] for c in common_cells], dtype=float)
valid = np.isfinite(pt_all)
common_cells = np.array(common_cells)[valid]
pt_all = pt_all[valid]

pt_norm_all = (pt_all - pt_all.min()) / (pt_all.max() - pt_all.min() + 1e-12)
order = np.argsort(pt_norm_all)
common_cells = common_cells[order]
pt_norm_all = pt_norm_all[order]

if len(common_cells) > N_CELLS:
    select_pos = np.linspace(0, len(common_cells)-1, N_CELLS).astype(int)
    selected_cells = common_cells[select_pos]
    pt0 = pt_norm_all[select_pos]
else:
    selected_cells = common_cells
    pt0 = pt_norm_all
N_CELLS = len(selected_cells)

dyn_df = pd.read_csv(DYNAMIC_GENES_CSV)
gene_col = next(c for c in dyn_df.columns if "gene" in c.lower() or "id" in c.lower())
score_cols = [c for c in dyn_df.columns if c != gene_col and pd.api.types.is_numeric_dtype(dyn_df[c])]
score_col = score_cols[0] if score_cols else None
ascending = any(k in score_col.lower() for k in ['pval', 'qval', 'p-value', 'q-value']) if score_col else False
if score_col:
    dyn_df = dyn_df.sort_values(score_col, ascending=ascending)

candidate_genes = dyn_df[gene_col].astype(str).str.strip().tolist()
selected_genes = [g for g in candidate_genes if g in counts_df.columns][:N_GENES]
if len(selected_genes) < N_GENES:
    extra = [g for g in counts_df.columns if g not in selected_genes]
    selected_genes += extra[:N_GENES - len(selected_genes)]
N_GENES = len(selected_genes)

Y0_raw = counts_df.loc[selected_cells, selected_genes].values.astype(np.float32)
Y0 = StandardScaler().fit_transform(np.log1p(Y0_raw)).astype(np.float32)
coords = np.array([all_coords[adata_index_map[c]] for c in selected_cells], dtype=np.float32)

np.save(SAVE_COORDS, coords)
np.save(SAVE_CELL_IDS, selected_cells)
np.save(SAVE_GENES, np.array(selected_genes, dtype=object))

nbrs = NearestNeighbors(n_neighbors=K_SPACE, metric='euclidean')
nbrs.fit(coords)
row, col = nbrs.kneighbors_graph(coords, mode='connectivity').nonzero()
edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long).to(device)
self_loop = torch.arange(N_CELLS, device=device)
edge_index = torch.cat([edge_index, torch.stack([self_loop, self_loop])], dim=1)

sort_idx = np.argsort(pt0)
pt_sorted = pt0[sort_idx]
Y_sorted = Y0[sort_idx]

def get_future_expression_interp(pt_current, pt_sorted, Y_sorted, step_size=STEP_SIZE):
    target_pt = np.clip(pt_current + step_size, 0, 1)
    indices = np.arange(len(pt_sorted))
    future_idx_float = np.interp(target_pt, pt_sorted, indices)
    future_idx_low = np.floor(future_idx_float).astype(int)
    future_idx_high = np.ceil(future_idx_float).astype(int)
    future_idx_high = np.clip(future_idx_high, 0, len(pt_sorted)-1)
    weight = future_idx_float - future_idx_low
    Y_future = (1 - weight)[:, None] * Y_sorted[future_idx_low] + weight[:, None] * Y_sorted[future_idx_high]
    return Y_future.astype(np.float32)

Y_targets = []
pt_targets = []
pt_current = pt0.copy()
for t in range(1, N_STEPS):
    Y_next = get_future_expression_interp(pt_current, pt_sorted, Y_sorted)
    pt_next = np.clip(pt_current + STEP_SIZE, 0, 1)
    Y_targets.append(Y_next)
    pt_targets.append(pt_next)
    pt_current = pt_next
Y_targets = np.stack(Y_targets, axis=0)
pt_targets = np.stack(pt_targets, axis=0)

class GraphGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        input_dim = in_channels + hidden_channels + 1
        self.conv_z = GATConv(input_dim, hidden_channels, heads=heads, concat=False)
        self.conv_r = GATConv(input_dim, hidden_channels, heads=heads, concat=False)
        self.conv_h = GATConv(input_dim, hidden_channels, heads=1, concat=False)
        self.out_proj_y = nn.Linear(hidden_channels, out_channels)
        self.out_proj_pt = nn.Linear(hidden_channels, 1)

    def forward(self, x, h, pt_cond, edge_index):
        combined = torch.cat([x, h, pt_cond], dim=-1)
        z = torch.sigmoid(self.conv_z(combined, edge_index))
        r = torch.sigmoid(self.conv_r(combined, edge_index))
        h_candidate_input = torch.cat([x, r * h, pt_cond], dim=-1)
        h_candidate = torch.tanh(self.conv_h(h_candidate_input, edge_index))
        h_next = (1 - z) * h + z * h_candidate
        delta_y = self.out_proj_y(h_next)
        pt_pred = torch.sigmoid(self.out_proj_pt(h_next))
        return delta_y, pt_pred, h_next

class GraphGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_steps, heads=2):
        super().__init__()
        self.cell = GraphGRUCell(in_channels, hidden_channels, out_channels, heads)
        self.n_steps = n_steps

    def forward(self, x0, pt_cond_seq, edge_index, teacher_forcing_seq_y=None, teacher_forcing_seq_pt=None, teacher_forcing_ratio=0.5):
        N = x0.size(0)
        h = torch.zeros(N, self.cell.hidden_channels, device=x0.device)
        y_seq = [x0]
        pt_seq = []
        for t in range(1, self.n_steps):
            if t == 1:
                pt_cond = pt_cond_seq[0]
            else:
                use_teacher = (teacher_forcing_seq_pt is not None and
                               t-2 < teacher_forcing_seq_pt.size(0) and
                               torch.rand(1).item() < teacher_forcing_ratio)
                if use_teacher:
                    pt_cond = teacher_forcing_seq_pt[t-2]
                else:
                    pt_cond = pt_seq[-1]
            if t == 1:
                x_input = x0
            else:
                use_teacher_y = (teacher_forcing_seq_y is not None and
                                 t-2 < teacher_forcing_seq_y.size(0) and
                                 torch.rand(1).item() < teacher_forcing_ratio)
                if use_teacher_y:
                    x_input = teacher_forcing_seq_y[t-2]
                else:
                    x_input = y_seq[-1]
            delta_y, pt_pred, h = self.cell(x_input, h, pt_cond, edge_index)
            y_next = x_input + delta_y
            y_seq.append(y_next)
            pt_seq.append(pt_pred)
        return torch.stack(y_seq, dim=0), torch.stack(pt_seq, dim=0)

model = GraphGRU(N_GENES, HIDDEN_DIM, N_GENES, N_STEPS, heads=HEADS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn_mse = nn.MSELoss()

x0 = torch.tensor(Y0, dtype=torch.float32, device=device)
pt_cond_seq = torch.tensor(pt_targets, dtype=torch.float32, device=device).unsqueeze(-1)
teacher_seq_y = torch.tensor(Y_targets, dtype=torch.float32, device=device)
teacher_seq_pt = torch.tensor(pt_targets, dtype=torch.float32, device=device).unsqueeze(-1)

indices = np.random.permutation(N_CELLS)
train_mask = torch.zeros(N_CELLS, dtype=torch.bool)
val_mask = torch.zeros(N_CELLS, dtype=torch.bool)
train_mask[indices[:int(0.8*N_CELLS)]] = True
val_mask[indices[int(0.8*N_CELLS):]] = True
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)

best_val_loss = float('inf')
for epoch in range(1, EPOCHS+1):
    model.train()
    optimizer.zero_grad()
    pred_y_seq, pred_pt_seq = model(x0, pt_cond_seq, edge_index,
                                    teacher_forcing_seq_y=teacher_seq_y,
                                    teacher_forcing_seq_pt=teacher_seq_pt,
                                    teacher_forcing_ratio=TEACHER_FORCING_RATIO)
    mse_loss = 0.0
    for t in range(1, N_STEPS):
        mse_loss += loss_fn_mse(pred_y_seq[t][train_mask], teacher_seq_y[t-1][train_mask])
    mse_loss = mse_loss / (N_STEPS - 1)
    pt_loss = 0.0
    for t in range(N_STEPS-1):
        pt_loss += loss_fn_mse(pred_pt_seq[t][train_mask], teacher_seq_pt[t][train_mask])
    pt_loss = pt_loss / (N_STEPS - 1)
    diversity_loss = 0.0
    for t in range(2, N_STEPS):
        diff = pred_y_seq[t] - pred_y_seq[t-1]
        step_diff = torch.norm(diff, dim=1).mean()
        step_diff_clamped = torch.clamp(step_diff, max=CLAMP_MAX)
        diversity_loss -= step_diff_clamped
    diversity_loss = diversity_loss / (N_STEPS - 1)
    mean_reg_loss = 0.0
    for t in range(1, N_STEPS):
        mean_reg_loss += torch.mean(pred_y_seq[t]) ** 2
    mean_reg_loss = mean_reg_loss / (N_STEPS - 1)
    total_loss = mse_loss + PT_LOSS_WEIGHT * pt_loss + DIVERSITY_LAMBDA * diversity_loss + MEAN_REG_LAMBDA * mean_reg_loss
    total_loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        val_y_seq, val_pt_seq = model(x0, pt_cond_seq, edge_index,
                                      teacher_forcing_seq_y=None,
                                      teacher_forcing_seq_pt=None,
                                      teacher_forcing_ratio=0.0)
        val_mse = 0.0
        for t in range(1, N_STEPS):
            val_mse += loss_fn_mse(val_y_seq[t][val_mask], teacher_seq_y[t-1][val_mask])
        val_mse = val_mse / (N_STEPS - 1)
        val_pt = 0.0
        for t in range(N_STEPS-1):
            val_pt += loss_fn_mse(val_pt_seq[t][val_mask], teacher_seq_pt[t][val_mask])
        val_pt = val_pt / (N_STEPS - 1)
        val_div = 0.0
        for t in range(2, N_STEPS):
            diff = val_y_seq[t] - val_y_seq[t-1]
            step_diff = torch.norm(diff, dim=1).mean()
            step_diff_clamped = torch.clamp(step_diff, max=CLAMP_MAX)
            val_div -= step_diff_clamped
        val_div = val_div / (N_STEPS - 1)
        val_mean_reg = 0.0
        for t in range(1, N_STEPS):
            val_mean_reg += torch.mean(val_y_seq[t]) ** 2
        val_mean_reg = val_mean_reg / (N_STEPS - 1)
        val_total = val_mse + PT_LOSS_WEIGHT * val_pt + DIVERSITY_LAMBDA * val_div + MEAN_REG_LAMBDA * val_mean_reg

        y_cpu = val_y_seq.cpu().numpy()
        pt_cpu = val_pt_seq.squeeze(-1).cpu().numpy()
        step_means_y = [y_cpu[t].mean() for t in range(N_STEPS)]
        step_means_pt = [pt0.mean()] + [pt_cpu[t].mean() for t in range(N_STEPS-1)]
        print(f"Epoch {epoch:03d}/{EPOCHS} | total_loss={total_loss.item():.4f} (mse={mse_loss.item():.4f}, pt={pt_loss.item():.4f}, div={diversity_loss.item():.4f}, mean_reg={mean_reg_loss.item():.4f}) | val_total={val_total.item():.4f}")
        print(f"  Y means: " + " ".join([f"t{t}:{m:.4f}" for t,m in enumerate(step_means_y)]))
        print(f"  PT means: " + " ".join([f"t{t}:{m:.4f}" for t,m in enumerate(step_means_pt)]))

    if val_total < best_val_loss:
        best_val_loss = val_total
        torch.save(model.state_dict(), MODEL_PATH)

    torch.cuda.empty_cache()

print("Training finished. Best val loss:", best_val_loss)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
with torch.no_grad():
    y_seq_full, pt_seq_full = model(x0, pt_cond_seq, edge_index,
                                    teacher_forcing_seq_y=None,
                                    teacher_forcing_seq_pt=None,
                                    teacher_forcing_ratio=0.0)
Y_series = y_seq_full.cpu().numpy()
PT_series_pred = pt_seq_full.squeeze(-1).cpu().numpy()
PT_series = np.vstack([pt0[np.newaxis, :], PT_series_pred])

for t in range(1, N_STEPS):
    PT_series[t] = np.maximum(PT_series[t], PT_series[t-1])

for t in range(1, N_STEPS):
    noise = np.random.normal(0, NOISE_STD, size=Y_series[t].shape)
    Y_series[t] = Y_series[t] + noise

np.save(SAVE_Y_DYNAMIC, Y_series.astype(np.float32))
np.save(SAVE_PT_DYNAMIC, PT_series.astype(np.float32))
print("Saved Y_dynamic:", SAVE_Y_DYNAMIC)
print("Saved PT_dynamic:", SAVE_PT_DYNAMIC)

for t in range(N_STEPS):
    print(f"t{t}: mean={PT_series[t].mean():.4f}, std={PT_series[t].std():.4f}")

colors = ['yellow', 'cyan', 'blue', 'darkviolet']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap), cax=ax, orientation='horizontal')
cb.set_label('Pseudotime')
plt.savefig(os.path.join(OUT_DIR, "custom_colormap_example.png"), dpi=150)
plt.close()
print("Custom colormap example saved.")