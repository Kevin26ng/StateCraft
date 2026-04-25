# Colab + Hugging Face + VS Code Setup

This project now includes cloud training via `scripts/train_cloud.py` and a Colab notebook at `notebooks/StateCraft_Colab_Train.ipynb`.

## 1) Security First

Do not commit tokens to git.
Use environment variables or Colab Secrets.

If you shared a token in chat or code, rotate it in Hugging Face settings.

## 2) Fast Path (Google Colab)

1. Open the notebook:
   `notebooks/StateCraft_Colab_Train.ipynb`
2. In Colab, choose Runtime -> Change runtime type -> GPU or TPU.
3. Add secret in Colab:
   - Name: `HF_TOKEN`
   - Value: your Hugging Face token
4. Run cells.
5. Replace `YOUR_USERNAME/statecraft-runs` with your real repo id.

## 3) VS Code Local Run + HF Upload

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set token in current terminal session:

```powershell
$env:HF_TOKEN = "your_hf_token_here"
```

Run cloud launcher with curriculum and push:

```powershell
python scripts/train_cloud.py --episodes 500 --scenario pandemic --curriculum --push-to-hub --hf-repo-id YOUR_USERNAME/statecraft-runs --hf-repo-type dataset
```

## 4) About GPU/TPU for this project

Current StateCraft logic is rule-based + NumPy-heavy and does not use a deep learning framework (like PyTorch/JAX) for model tensor training loops.

That means GPU/TPU acceleration is limited right now.
The fastest practical gains today are:
- Colab high-CPU runtime
- Running longer jobs in one session
- Saving and comparing many runs via Hugging Face artifacts

## 5) Hugging Face credits best use

With your credits, best value for this repository is:
- Store run artifacts/metrics as datasets
- Track experiments by run folders (`outputs/run_YYYYMMDD_HHMMSS`)
- Keep private repos for sensitive runs (`--hf-private`)

## 6) Optional commands

Private dataset upload:

```powershell
python scripts/train_cloud.py --episodes 500 --curriculum --push-to-hub --hf-repo-id YOUR_USERNAME/statecraft-private-runs --hf-repo-type dataset --hf-private
```

No upload, local artifacts only:

```powershell
python scripts/train_cloud.py --episodes 300 --scenario economic --curriculum
```
