# CO5085 — Deep Learning and Its Applications

Coursework repository of **Tieu Tri Kiet**: 4 exercises (E1–E4) and 2 assignments (A1–A2),
all implemented in PyTorch and backed by real empirical experiments.

📄 **Landing page:** https://tieutrikiet.github.io/co5085_tieutrikiet_dl_cv/

## Contents

| Dir | Topic | Deliverable |
| --- | --- | --- |
| `E1/` | Image classification on a small image set | Softmax vs. MLP vs. CNN — accuracy, capacity, error cases |
| `E2/` | Multi-Head Self-Attention | From scratch + `torch.nn`, several tokenizations (patch, row, CNN-stem, CLS) |
| `E3/` | Sequence models | LSTM and GRU over pixel/row sequences vs. a feed-forward / CNN baseline |
| `E4/` | Generative models | VAE, GAN, Diffusion — samples, latent space, quality vs. compute |
| `A1/` | CNN vs. Transformer on a large image dataset | A1.1 dataset + protocol, A1.2 final comparison |
| `A2/` | Self-chosen topic + paper analysis | A2.1 topic/paper/data, A2.2 implementation + write-up |

Each directory holds its own `description.md` (the brief), notebooks, and saved results.
E1–E3 and E4's dataset reuse the **same** small image set so results stay comparable.

## Setup

A shared virtualenv lives one level up at `../.venv` (Python 3.11 via pyenv — the system
Python 3.14 is too new for current PyTorch wheels):

```bash
source ../.venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon `torch.backends.mps.is_available()` is `True`, so prefer the `mps` device
over `cpu` for training.

## Datasets

Datasets are **not** committed — they live outside the repo (one level up in `CO5085/`) and
are gitignored. Expected layout:

```
CO5085/
├── MNIST/  Fashion-MNIST/  cifar-10/  food-101/   ← data, not in git
└── co5085_tieutrikiet_dl_cv/                      ← this repo
```

Point notebooks at `../` (or a `DATA_ROOT` variable) rather than hard-coding absolute paths.

## GitHub Pages

The landing page is `index.html` at the repo root, deployed by
`.github/workflows/pages.yml` on every push to `main`.
