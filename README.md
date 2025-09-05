# Verifying Out-of-Distribution Robustness in Multi-Spectral Change Detection  
Code release for the ICLR 2025 paper (anonymous submission).  

This repository contains training, verification, and OOD evaluation notebooks for multi-spectral change detection models under sensor-calibrated perturbations.  

---

## 📂 Repository Structure
- **hct_novel_training.ipynb** — Head-Consistency Training (HCT) recipe  
- **Verifier_v13_withHCT.ipynb** — Tail-tapped verifier (α/β-CROWN diagnostic)  
- **croprot_ood_dataset_collection.ipynb** — CropRot dataset curation (NDVI differencing + QA)  
- **croprot_ood_dataset_exploration.ipynb** — Dataset sanity checks and statistics  
- **croprot_ood_eval_oscd.ipynb** — OOD evaluation on OSCD and CropRot  
- **ood_test-synthetic.ipynb** — Synthetic perturbation benchmarks (drift, shadow, blur, passband)  
- **original_oscd_nets.ipynb** — Baseline OSCD backbones  
- **standalone_croprot_nets.ipynb** — CropRot-trained backbones  
- **models.py** — Backbone definitions (FresUNet, FALCONet, AttU-Net)  
- **\*.pth.tar** — Pretrained checkpoints (verification-friendly encoder–decoders)  
- **req.txt** — Python dependencies  

---

## 🛠️ Setup Instructions

We recommend using a **Python virtual environment** (tested with Python 3.12.3).  

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# On Windows (PowerShell):
# venv\Scripts\Activate.ps1

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install requirements
pip install -r req.txt
```

---

## 🚀 Usage

1. Launch Jupyter Lab or Notebook:  
   ```bash
   jupyter lab
   ```  
   or  
   ```bash
   jupyter notebook
   ```

2. Open the relevant notebook:  
   - Train a model with **`hct_novel_training.ipynb`**  
   - Run certification diagnostics with **`Verifier_v13_withHCT.ipynb`**  
   - Reproduce OOD results with **`croprot_ood_eval_oscd.ipynb`**  

3. Pretrained models (`.pth.tar`) can be loaded directly via `torch.load`.

---

## ⚡ Quick Start Example

To quickly sanity-check installation, load a pretrained FALCONet model and run a forward pass on dummy inputs:

```python
import torch
from models import FALCONet

# Load pretrained model
model = FALCONet()
state_dict = torch.load("FALCONetMHA_LiRPA.pth.tar", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Dummy Sentinel-2 pair (2 images × 13 bands = 26 channels)
x = torch.randn(1, 26, 128, 128)
with torch.no_grad():
    logits = model(x)

print("Logits shape:", logits.shape)  # Expected: [1, 2, 128, 128]
```

---

## 📑 Notes
- Certification experiments run on CPU; HCT training benefits from GPU but is lightweight enough for CPU.  
- CropRot dataset curation requires internet access (SentinelHub API). See **`croprot_ood_dataset_collection.ipynb`** for details.  
- All perturbation families (drift, shadow, blur, passband) are implemented inside the training and testing notebooks.  

---

## 📜 Citation
TBD
