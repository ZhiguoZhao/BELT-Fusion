# BELT-Fusion: Bayesian Evidential Late Fusion for Trustworthy V2X Perception

<img src="https://img.shields.io/badge/Code%20Status-Coming%20Soon-important" alt="Code Status"> [![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/XXXX.XXXX)

## 🚀 Overview
**BELT-Fusion** is a probabilistic framework designed to address critical uncertainty challenges in Vehicle-to-Everything (V2X) collaborative perception. By enabling trustworthy late fusion, our method significantly improves 3D object detection robustness under real-world noisy conditions (e.g., localization errors, asynchronous data, and heterogeneous agents), achieving **7.16% AP@0.7 improvement** in noisy environments compared to uncertainty-agnostic baselines.

## 🔑 Key Features
### 🧠 Intelligent Uncertainty Modeling
- **Dual-path Uncertainty Decoupling**  
  ✓ Classification uncertainty via *Evidential Deep Learning*  
  ✓ Regression uncertainty via *Bayesian Neural Networks*  
  ✓ Plug-and-play compatibility with existing object detectors

### ⚡ Dynamic Fusion Engine
- **Uncertainty-aware Weight Allocation**  
  ✓ Real-time quantification of fusion-level reliability  
  ✓ Adaptive object selection and confidence weighting  
  ✓ Zero-retraining deployment (plug-and-play)

## 📊 Performance Highlights
| Scenario           | Dataset       | AP@0.7 Improvement |
|---------------------|---------------|--------------------|
| Noisy Environment   | OPV2V + DAIR-V2X       | **+7.16%**         |
| Ideal Conditions    | OPV2V + DAIR-V2X      | **+3.84%**         |

## 🛠️ Implementation Details
- **Supported Tasks**: 3D Object Detection
- **Tested Datasets**: 
  - [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) (Real-world)
  - [DAIR-V2X](https://thudair.baai.ac.cn/index) (Simulation)
- **Runtime Compatibility**: PyTorch-based implementation

## 📦 Code Availability
We are actively preparing a well-documented and modular codebase. Expected release phases:

```markdown
[!NOTE]  
Code Release Timeline:  
✓ Core framework: Q3 2024  
✓ Pre-trained models: Q3 2024  
✓ Tutorials & Demos: Q4 2024
```

## 📜 Citation
```bibtex
@article{beltfusion2024,
  title={BELT-Fusion: Trustworthy V2X Collaborative Perception via Uncertainty-Aware Late Fusion},
  author={Author, A. and Coauthor, B.},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2024}
}
```

## 💡 Why BELT-Fusion Matters
Traditional V2X fusion methods suffer from:  
⚠️ Blind fusion of unreliable observations  
⚠️ Performance degradation under real-world noise  
⚠️ Lack of uncertainty quantification  

Our solution provides:  
✅ Quantifiable reliability metrics for each detection  
✅ Noise-robust fusion decisions  
✅ Backward compatibility with existing detectors

---

[![Star](https://img.shields.io/github/stars/yourusername/BELT-Fusion?style=social)](https://github.com/yourusername/BELT-Fusion)  [![Watch](https://img.shields.io/github/watchers/yourusername/BELT-Fusion?style=social)](https://github.com/yourusername/BELT-Fusion)

**For collaboration inquiries:** 2410805@tongji.edu.cn
