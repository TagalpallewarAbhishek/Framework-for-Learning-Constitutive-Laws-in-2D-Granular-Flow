# Neural Network Frameworks for Constitutive Laws in 2D Granular Flow

**Author:** Abhishek Tagalpallewar  
**Supervisor:** Harkirat Singh  


## 📋 Executive Summary

This repository archives a machine learning research initiative aimed at **discovering the underlying physical laws governing granular materials through neural network function approximation**. The core objective is to map **kinematic velocity gradients to kinetic stress tensors** in 2D granular flows using deep learning, with a focus on:

- Developing scale-invariant constitutive law models
- Systematic architectural optimization via hyperparameter tournaments
- Cross-dataset generalization and robustness validation
- Identifying hidden coupling mechanisms through zero-gravity testing

The research demonstrates that carefully trained neural networks can capture the non-linear physics of granular materials and generalize across different simulation domains and experimental conditions.

---

## 📂 Repository Structure

### **Code & Notebooks (Interactive Research)**

```
.
├── Phase_1_Math_Foundations.ipynb          # 🧮 Sanity Check: Function Approximation
├── Phase_2_Architectural_Tournament.ipynb   # 🏆 Hyperparameter Optimization (75 experiments)
├── Granular_Constitutive_Discovery.ipynb   # 🚀 Advanced Pipeline: Training & Validation
├── Report.pdf                               # 📄 Final Technical Report (4 pages)
├── Midsemester_presentation.pdf             # 📊 Mid-Semester Progress Presentation
├── Endsemester_presentation.pdf             # 📈 End-Semester Results & A1 Poster
└── README.md                                # This file
```

### **Phase Descriptions**

#### **Phase 1: Mathematical Foundations** (`Phase_1_Math_Foundations.ipynb`)
- **Objective:** Validate that feedforward neural networks can approximate non-linear periodic functions
- **Method:** Train a 3-layer network (1→20→20→1) with Tanh activation on sine wave data
- **Outcome:** Demonstrates interpolation vs. extrapolation capability and establishes baseline architecture
- **Key Concept:** Serves as the "sanity check" before deploying on real granular DEM data
- **Loss Metric:** Relative Error (RE) for scale-invariant evaluation

#### **Phase 2: Architectural Tournament** (`Phase_2_Architectural_Tournament.ipynb`)
- **Objective:** Systematically benchmark **75 unique combinations** across:
  - **5 Activation Functions:** ReLU, Tanh, Swish, Mish, GELU
  - **5 Loss Functions:** MSE, MAE, Huber, MSLE, Relative Error (RE)
  - **3 Physical Domains:** Exponential (energy dissipation), Logarithmic (asymptotic pressure), Fourier (complex oscillations/force chains)
- **Architecture:** Standardized 3-layer network (64 nodes) identified as optimal capacity-to-speed ratio
- **Result:** Identifies "champion" combinations (e.g., **Tanh + Huber** or **Mish + Relative Error**)
- **Optimization:** AdamW optimizer with Cosine Annealing learning rate scheduling

#### **Phase 3: Advanced Constitutive Training & Validation** (`Granular_Constitutive_Discovery.ipynb`)

**🔹 Training Pipeline:**
- Loads LAMMPS DEM simulation data from `/home/abhishek/granular_data`
- **Horizontal Averaging:** Reduces particle-level noise to bulk flow trends
- **Quasi-Linear Masking (QLS):** Isolates stable flow regions by detecting regions with low local slope variance
- **Wall Masking:** Filters boundary layer interference (|∇v| > 0.3)
- **Breakthrough: Reference Normalization** → Scales inputs/outputs by dataset means to enable cross-dataset learning

**🔹 Data Augmentation:**
- Gaussian noise injection (σ=0.01) to double training dataset size
- StandardScaler applied post-augmentation for numerical stability

**🔹 Tournament Architecture:**
- Extended network: 1 → 128 → 64 → 3 (predicts σ_xx, σ_xy, σ_yy)
- Champion selection via **Relative Error on validation set**
- Trains 25 model combinations (5 activations × 5 losses) for 3000 epochs

**🔹 Validation Set 1 - Cross-Dataset Generalization:**
- **Test:** Deploy frozen champion model on `granular_data3` (different simulation)
- **Breakthrough:** Reference Normalization reduces initial error from **48% → 3%**
- **Interpretation:** Proves the model learned the "Physics of the Law" rather than just numeric correlation
- **Ensemble Strategy:** Top-5 models averaged for maximum consensus
- **Key Result:** Scale-invariance confirmed across datasets

**🔹 Validation Set 2 - Zero-Gravity Bulk Flow Analysis:**
- **Test:** Deploy model on zero-gravity bulk flow simulations (boundary-driven, not gravity-driven)
- **Discovery: Pressure-Source Decoupling**
  - Normal stress predictions remain accurate (RE ≈ 3-5%)
  - Shear stress predictions diverge to **RE ≈ 84%**
  - **Interpretation:** Model implicitly learned a correlation between depth and pressure during gravity-influenced training
  - **Physical Insight:** This correlation does NOT exist in boundary-driven bulk flows
  - **Research Value:** Identifies hidden model dependencies and suggests need for depth-normalized architectures

---

## 🔬 Key Research Breakthroughs

### **1. Reference Normalization (Scale Invariance)**

**Problem:** Deploying a model trained on one dataset to another with different magnitude scales yields **~48% relative error**.

**Solution:** Scale both inputs and outputs by their respective dataset means before normalization:
```
X_scaled = X_raw / mean(X_raw)
Y_scaled = Y_raw / mean(Y_raw)
```

**Result:** Cross-dataset error drops to **~3%**, proving the network learned scale-invariant physics laws.

**Impact:** Enables transfer learning across different simulation configurations without retraining.

### **2. Pressure-Source Decoupling**

**Objective:** Validate model performance under radically different boundary conditions (zero gravity).

**Finding:** When deployed on zero-gravity bulk flows:
- Normal stresses (σ_xx, σ_yy) remain accurate (RE < 5%)
- Shear stress (σ_xy) error diverges to ~84%

**Interpretation:** 
- The model learned an **implicit correlation between depth (z) and normal pressure** during gravity-driven training
- This correlation does NOT exist in boundary-driven bulk flows
- Demonstrates that neural networks can capture hidden physical couplings but may not generalize to fundamentally different regimes

**Research Implication:** Future work should explore depth-normalized architectures or multi-task learning to disentangle pressure sources.

### **3. Quasi-Linear Masking (Data Quality Control)**

**Method:** Identifies stable flow regions by computing local slope variance over a rolling window.
```
local_std = rolling_std(gradient(profile), window=25)
mask = local_std < percentile(local_std, 60)
```

**Benefit:**
- Removes transitional/unstable regions with high noise
- Focuses training on physically meaningful (linear constitutive) regions
- Improves data signal-to-noise ratio by filtering spurious outliers

---

## 🏆 Model Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Champion Architecture** | Tanh/Mish + Huber/RE | Identified via 75-experiment tournament |
| **Training Samples** | ~600 pixels × 2 (with augmentation) | Horizontal averaging reduces noise |
| **Cross-Dataset Error** | 3% (Reference Normalized) | Validates scale invariance |
| **Zero-Gravity Normal Stresses** | 3-5% | Robust to gravity removal |
| **Zero-Gravity Shear Stress** | 84% | Reveals depth-pressure coupling |
| **Optimization** | AdamW + Cosine Annealing | 3000 iterations per model |

---

## 🛠️ Technical Requirements

### **Core Dependencies**
- **Python 3.7+**
- **PyTorch** (≥1.9.0) - Deep learning framework
- **NumPy** (≥1.19.0) - Numerical computing
- **Pandas** (≥1.1.0) - Data manipulation and analysis
- **Matplotlib** (≥3.3.0) - Visualization
- **Scikit-Learn** (≥0.24.0) - StandardScaler, train_test_split
- **Jupyter Notebook** - Interactive development environment

### **Data Generation (Optional)**
- **LAMMPS** (≥2019) - Granular dynamics simulation
- **dumpatom_parser** or custom script - Post-processing DEM output

### **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd ME299

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy pandas matplotlib scikit-learn jupyter

# Launch Jupyter
jupyter notebook
```

---

## 📖 How to Use This Repository

### **For Quick Understanding: Read in Order**
1. **Phase_1_Math_Foundations.ipynb** → Understand the NN basics and architecture choice
2. **Phase_2_Architectural_Tournament.ipynb** → See the systematic hyperparameter search
3. **Granular_Constitutive_Discovery.ipynb** → Deep dive into the full pipeline and validations

### **For Reproducibility**
1. Ensure you have LAMMPS simulation data in `/home/abhishek/granular_data`, `/home/abhishek/granular_data3`, and `/home/abhishek/bulk_sim_data`
2. Run all cells in `Granular_Constitutive_Discovery.ipynb` sequentially
3. Compare your validation plots with the Reference Normalization and Zero-Gravity sections

### **For Extending the Work**
- **Modify activations/losses:** Edit the `act_map` and `get_loss_fn()` dictionaries in Phase 2 or 3
- **Adjust network depth:** Change the `nn.Sequential()` definition
- **Test new data:** Update file paths and apply Reference Normalization with learned scale ratios
- **Investigate decoupling:** Add depth-normalized input features (z/H) to address pressure-source issue

---

## 📊 Documentation & Presentations

### **Final Technical Report** ([Report.pdf](Report.pdf))
- **Content:** 4-page comprehensive technical breakdown
- **Sections:** 
  - Methodology and mathematical foundations
  - Reference Normalization scaling breakthroughs
  - Architectural tournament results and champion selection
  - Physical interpretations and constitutive law discovery
  - Validation results and scaling assumptions
- **Audience:** Academic researchers, peer review

### **Mid-Semester Presentation** ([Midsemester_presentation.pdf](Midsemester_presentation.pdf))
- **Content:** Preliminary progress report
- **Highlights:**
  - Initial study on function approximation (sine waves)
  - 3-layer architecture selection rationale
  - Early experimental designs
- **Audience:** Professor and research committee

### **End-Semester Poster** ([Endsemester_presentation.pdf](Endsemester_presentation.pdf))
- **Content:** A1 poster format for research symposium
- **Focus:**
  - Scale-invariance via Reference Normalization
  - Pressure-Source Decoupling phenomenon
  - Zero-gravity validation results
  - Key research insights and future directions
- **Audience:** General research community, symposium attendees

---

## 🔍 Interpreting Results

### **Relative Error (RE) Metric**

The primary evaluation metric throughout is **Relative Error**:

$$\text{RE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|\hat{y}_i - y_i|}{|y_i| + \epsilon}$$

- **Scale-invariant:** Normalizes by true value, making it independent of absolute magnitudes
- **Interpretable:** 3% RE means predictions are off by a factor that's 3% of the true value
- **Disadvantage:** Sensitive when |y| is small (mitigated by ε = 1e-8)

### **Tournament Leaderboards**

Each phase outputs a ranked leaderboard of model combinations sorted by **Final Relative Error**. The "champion" model balances:
- ✅ Low validation error
- ✅ Computational efficiency
- ✅ Physical interpretability
- ✅ Generalization across domains

---

## 🚀 Key Insights & Future Work

### **Current Achievements**
1. ✅ Demonstrated NN capability to capture granular constitutive laws
2. ✅ Achieved near-universal scale invariance (3% cross-dataset error)
3. ✅ Identified hidden physical couplings (pressure-depth correlation)
4. ✅ Systematic hyperparameter optimization framework

### **Future Directions**
1. **Depth Normalization:** Add z/H as input feature to decouple pressure sources
2. **Multi-Task Learning:** Separately predict normal vs. shear stress with different architectures
3. **Uncertainty Quantification:** Add Bayesian layers to quantify model confidence
4. **3D Extension:** Generalize from 2D velocity gradients to full 3D stress tensors
5. **Real Experimental Data:** Validate against physical granular experiments (not just simulations)
6. **Physics-Informed Loss:** Incorporate conservation laws (momentum, energy) into training
7. **Sparse Inference:** Deploy on edge devices for real-time simulation acceleration

---

## 📝 Citation & Attribution

If you use this repository, please cite:

```bibtex
@research{Tagalpallewar2024GranularNN,
  author = {Tagalpallewar, Abhishek},
  title = {Neural Network Frameworks for Constitutive Laws in 2D Granular Flow},
  institution = {IIT Gandhinagar},
  year = {2024},
  supervisor = {Harkirat Singh},
  type = {Graduate Research Project}
}
```

---

## 📧 Contact & Support

**Author:** Abhishek Tagalpallewar  
**Institution:** Indian Institute of Technology Gandhinagar  
**Advisor:** Prof. Harkirat Singh  

For questions, clarifications, or extensions of this work, please reach out or submit an issue to the repository.

---

## 📜 License

[Specify your preferred license here - e.g., MIT, Apache 2.0, or Custom Academic License]

---

## 🙏 Acknowledgments

- **LAMMPS Community:** For robust granular dynamics simulation framework
- **PyTorch Team:** For flexible deep learning tools
- **IIT Gandhinagar:** For computational resources and academic support
- **Supervisor Prof. Harkirat Singh:** For guidance, feedback, and domain expertise

---

## 📌 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | May 2024 | Initial repository documentation with Phase 1-3 pipeline |
| - | - | Added comprehensive section on Reference Normalization |
| - | - | Documented Pressure-Source Decoupling discovery |

---

**Last Updated:** May 1, 2026  
**Status:** Research Complete - Open for Collaboration and Extensions
