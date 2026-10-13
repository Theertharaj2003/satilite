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

from src.distillation.models import TeacherNet, StudentNet
from src.distillation.loss import DistillationLoss


class RandomDemoDataset(Dataset):
    def __init__(self, n=128, classes=3):
        self.x = torch.rand(n, 3, 64, 64)
        self.y = torch.randint(0, classes, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--output", default="outputs/checkpoints/student_demo.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = TeacherNet().to(device).eval()
    student = StudentNet().to(device).train()
    # Teacher is frozen for this starter experiment.
    for p in teacher.parameters():
        p.requires_grad = False
    loss_fn = DistillationLoss()
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3)
    loader = DataLoader(RandomDemoDataset(), batch_size=16, shuffle=True)
    for epoch in range(args.epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss = loss_fn(student_logits, teacher_logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch+1} loss={total/len(loader):.6f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out)
    print(f"Saved student prototype: {out}")


if __name__ == "__main__":
    main()
