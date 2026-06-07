"""Script to train a simple Mapping Prior from 512-dim ViT-B/32 to 1024-dim ViT-H/14."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class MappingPrior(nn.Module):
    def __init__(self, in_dim=512, out_dim=1024, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    print("Loading 512-dim source embeddings...")
    z_512 = np.load("data/processed/clip_image_emb.npy")
    z_512 = torch.from_numpy(z_512).float()
    
    z_1024_path = Path("data/processed/clip_image_emb_1024.npy")
    if z_1024_path.exists():
        print("Loading cached 1024-dim target embeddings...")
        z_1024 = torch.from_numpy(np.load(z_1024_path)).float()
    else:
        print("Computing 1024-dim target embeddings for mapping...")
        paths = json.loads(Path("data/processed/clip_image_paths.json").read_text())
        
        model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").eval()
        processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        
        z_1024_list = []
        batch_size = 32
        for i in tqdm(range(0, len(paths), batch_size)):
            batch_paths = paths[i:i+batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                z_1024_list.append(feats)
        z_1024 = torch.cat(z_1024_list, dim=0)
        
        print("Saving 1024-dim target embeddings for future runs...")
        np.save(z_1024_path, z_1024.cpu().numpy())
    
    print("Splitting into Train (1800) and Val (196)...")
    indices = np.random.RandomState(42).permutation(len(z_512))
    train_idx, val_idx = indices[:1800], indices[1800:]
    
    train_z512, val_z512 = z_512[train_idx], z_512[val_idx]
    train_z1024, val_z1024 = z_1024[train_idx], z_1024[val_idx]
    
    dataset = TensorDataset(train_z512, train_z1024)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    prior = MappingPrior()
    optimizer = torch.optim.AdamW(prior.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    
    print("Training Mapping Prior (512 -> 1024) with Validation...")
    for epoch in range(300):
        prior.train()
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = prior(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            prior.eval()
            with torch.no_grad():
                val_pred = prior(val_z512)
                val_mse = F.mse_loss(val_pred, val_z1024).item()
                
                # Compute retrieval metrics (Cosine Similarity)
                # Normalize just for retrieval testing
                pred_norm = F.normalize(val_pred, dim=-1)
                targ_norm = F.normalize(z_1024, dim=-1) # against the FULL bank
                
                sims = pred_norm @ targ_norm.T
                cos_mean = (F.normalize(val_pred, dim=-1) * F.normalize(val_z1024, dim=-1)).sum(dim=-1).mean().item()
                
                top1 = 0
                for i, true_idx in enumerate(val_idx):
                    rank = (sims[i] >= sims[i, true_idx]).sum().item()
                    if rank == 1:
                        top1 += 1
                top1_acc = top1 / len(val_idx)
                
            print(f"Epoch {epoch+1:03d} | Train MSE: {total_loss/len(loader):.4f} | Val MSE: {val_mse:.4f} | Val Cosine: {cos_mean:.3f} | Val Top-1: {top1_acc*100:.1f}%")
            
    print("Saving Mapping Prior to weights/mapping_prior.pt...")
    Path("weights").mkdir(exist_ok=True)
    # Save using the full dataset eventually, but for now just save this one
    torch.save(prior.state_dict(), "weights/mapping_prior.pt")
    print("Done!")

if __name__ == "__main__":
    main()
