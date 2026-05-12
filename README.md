# GateKD: Confidence-Gated Closed-Loop Distillation for Robust Reasoning

<div align="center">

# 🧠 GateKD

### Confidence-Gated Closed-Loop Distillation for Robust Reasoning

[![ACL](https://img.shields.io/badge/ACL-2026-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red.svg)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>

---

## 🚀 Overview

GateKD is a confidence-gated reasoning distillation framework designed for transferring robust multi-step reasoning abilities from large language models into compact student models.

Unlike traditional open-loop knowledge distillation approaches, GateKD introduces:

- ✅ Confidence-Gated Soft Supervision
- ✅ Gated Hidden-State Evolution
- ✅ Reliability-Filtered Attention Distillation
- ✅ Closed-Loop Reasoning Transfer

The framework dynamically suppresses unreliable teacher supervision while reinforcing stable reasoning trajectories.

---

## 🏆 Accepted Paper

**GateKD: Confidence-Gated Closed-Loop Distillation for Robust Reasoning**

Accepted to:
- ACL 2026 Workshop TrustNLP Fast Track

---

# 📦 Repository Structure

```bash
GateKD/
├── configs/
│   ├── t5_small.yaml
│   └── flant5_small.yaml
├── datasets/
│   └── prepare_data.py
├── gatekd/
│   ├── modeling/
│   │   ├── gatekd_model.py
│   │   ├── losses.py
│   │   └── confidence.py
│   ├── trainer/
│   │   └── trainer_gatekd.py
│   └── utils/
│       ├── metrics.py
│       └── seed.py
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

# ⚡ Installation

```bash
git clone https://github.com/kaopanboonyuen/GateKD.git
cd GateKD

pip install -r requirements.txt
```

---

# 🔥 Training

## T5-small

```bash
bash scripts/train.sh
```

Or manually:

```bash
python train.py \
    --config configs/t5_small.yaml
```

---

# 📊 Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/gatekd_t5_small.pt
```

---

# 🧠 GateKD Objective

The final training objective:

L = L_task + λ1 * L_gate_soft + λ2 * L_gate_hidden + λ3 * L_gate_attention

GateKD dynamically gates:
- teacher logits
- hidden states
- attention maps

based on predictive entropy confidence.

---

# 📈 Supported Benchmarks

| Task | Category |
|---|---|
| CSQA | Commonsense |
| StrategyQA | Commonsense |
| Shuffled Objects | Logical |
| Last Letter | Symbolic |

---

# 🔬 Features

- HuggingFace Transformers integration
- Multi-GPU support
- FP16 mixed precision
- Deterministic training
- Reproducible ACL-ready experiments
- Modular distillation pipeline

---

# 📜 Citation

```bibtex
@inproceedings{kao2026gatekd,
  title={GateKD: Confidence-Gated Closed-Loop Distillation for Robust Reasoning},
  author={Sermsri, K. and Panboonyuen, T.},
  booktitle={ACL 2026 Workshop TrustNLP Fast Track},
  year={2026}
}
```

---

# ✨ Corresponding Author

**Teerapong Panboonyuen** served as the corresponding author and principal contributor of this work, leading the research direction, conceptual development, methodology design, implementation, experimentation, and scientific analysis throughout the project.

---

# ❤️ Acknowledgements

Developed at:
- Chulalongkorn University
- MARSAIL (Motor AI Recognition Solution AI Laboratory)
