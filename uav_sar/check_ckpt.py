import torch
ckpt = torch.load("checkpoints/latest.pt", map_location="cpu")
print("Epoch in latest.pt:", ckpt.get("epoch"))
print("Curriculum state in latest.pt:", ckpt.get("curriculum_state"))
for i in range(4):
    key = f"agent_{i}"
    if key in ckpt:
        print(f"Agent {i} epsilon:", ckpt[key].get("epsilon"))
