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
        h = self.net(x)
        return F.normalize(h, dim=-1)

def main():
    print("Loading 512-dim source embeddings...")
    z_512 = np.load("data/processed/clip_image_emb.npy")
    z_512 = torch.from_numpy(z_512).float()
    
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
            feats = F.normalize(feats, dim=-1)
            z_1024_list.append(feats)
    z_1024 = torch.cat(z_1024_list, dim=0)
    
    print("Training Mapping Prior (512 -> 1024)...")
    dataset = TensorDataset(z_512, z_1024)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    prior = MappingPrior()
    optimizer = torch.optim.AdamW(prior.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(100):
        prior.train()
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = prior(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, MSE Loss: {total_loss/len(loader):.4f}")
            
    print("Saving Mapping Prior to weights/mapping_prior.pt...")
    Path("weights").mkdir(exist_ok=True)
    torch.save(prior.state_dict(), "weights/mapping_prior.pt")
    print("Done! The generator can now natively translate your 512-dim predictions to 1024-dim IP-Adapter inputs.")

if __name__ == "__main__":
    main()
