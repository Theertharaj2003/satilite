from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import cv2

from src.data.dataset_loader import PairedEODataset
from src.fusion.fusion_model import CrossModalFusionNet, preprocess_image, save_checkpoint
from src.fusion.losses import FusionReconstructionLoss


class TorchPairDataset(Dataset):
    def __init__(self, root_eo, root_ir, size):
        self.base = PairedEODataset(root_eo, root_ir)
        self.size = size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        eo = preprocess_image(item["eo"], self.size)[0]
        ir = preprocess_image(item["ir"], self.size)[0]
        return eo, ir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eo-dir", default="data/raw/eo")
    ap.add_argument("--ir-dir", default="data/raw/ir")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--output", default="outputs/checkpoints/fusion_best.pt")
    args = ap.parse_args()
    ds = TorchPairDataset(args.eo_dir, args.ir_dir, args.size)
    if len(ds) == 0:
        raise SystemExit("No paired EO/IR images found. Use tools/generate_demo_data.py first or add a real paired dataset.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossModalFusionNet().to(device)
    loss_fn = FusionReconstructionLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for eo, ir in loader:
            eo, ir = eo.to(device), ir.to(device)
            out = model(eo, ir)
            loss = loss_fn(out, eo, ir)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
        mean_loss = running / max(1, len(loader))
        print(f"epoch={epoch+1}/{args.epochs} loss={mean_loss:.6f}")
        if mean_loss < best:
            best = mean_loss
            save_checkpoint(model, args.output)
    print(f"Best checkpoint: {args.output}")


if __name__ == "__main__":
    main()
