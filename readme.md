# Deep3DCCS

**Deep Learning Approach for CCS Prediction from Multi-Angle Projections of Molecular Geometry**

---

## Overview

**Deep3DCCS** is a comprehensive deep learning framework for predicting **Collision Cross-Section (CCS)** values using three-dimensional molecular structure information derived from mass spectrometry data. The framework employs a specialized **3D Convolutional Neural Network (3DCNN)** operating on **multi-angle 2D projections** of molecular geometries. This design bridges molecular structure representation with ion mobility spectrometry, enabling accurate CCS prediction for applications in **metabolomics, lipidomics, and structural biology**.

---

## Key Features

### Molecular Structure Processing
- Converts **SMILES** into optimized **3D molecular structures**
- Utilizes **Gaussian View 6.0 / Gaussian 16**, **RDKit**, and computational chemistry methods
- Generates physically realistic geometries suitable for CCS modeling

### Multi-View 2D Projections
- Generates rotational 2D projections around **x-, y-, and z-axes**
- Configurable rotational increments (1, 3, 5, or 10 angles per axis)
- Image resolutions from **8×8 to 192×192 pixels**
- Center-of-mass recentering and van der Waals surface visualization

### 3DCNN Regression Model
- Custom **3D CNN architecture** optimized for CCS regression
- Processes volumetric stacks of multi-angle projections
- Includes 3D convolution blocks, global average pooling, and dense regression layers

### Automated Workflow (GUI)
- **PyQt5-based graphical interface**
- End-to-end pipeline from SMILES input to CCS prediction
- Modular workflow tabs:
  - Molecular Optimization
  - 2D Projection Generation
  - Model Training
  - Inference

### Comprehensive Evaluation
- Primary metric: **Relative Percentage Error (RPE)**
- Additional metrics: MAE, MAPE, Pearson r, R², standard error
- Automated visualizations:
  - Scatter plots
  - Residual density plots
  - QQ plots
  - Comparative bar charts

---

## Installation

### Prerequisites
- **Python ≥ 3.7** (tested with TensorFlow 2.10)
- Supported OS: **Windows 10/11**, **Ubuntu 20.04/22.04**

### Hardware Recommendations
- **GPU (recommended):** NVIDIA CUDA-compatible GPU (≥ 8 GB VRAM)
- **RAM:** ≥ 16 GB
- CPU-only execution supported for small datasets

### Clone Repository
```bash
git clone https://github.com/yourusername/Deep3DCCS.git
cd Deep3DCCS
```

### Core Dependencies
```bash
pip install tensorflow==2.10.0
pip install rdkit-pypi
pip install pyqt5 matplotlib numpy pandas scipy scikit-learn opencv-python pillow seaborn tqdm colorama termcolor
```

### RDKit (Alternative via Conda)
```bash
conda install -c conda-forge rdkit
conda install -c conda-forge openbabel
```

### Installation Verification
```bash
python -c "import tensorflow as tf; import rdkit; print('Installation successful:', tf.__version__)"
```

> A standalone **portable Windows (64-bit)** version will be released in the future.

---

## Usage

### GUI Interface
Launch the main application:
```bash
python 3DCNN_gui.py
```

The GUI provides:
- **Molecular Optimization:** SMILES → optimized 3D structures
- **2D Projection Generation:** Multi-angle image stacks
- **Model Training:** Architecture and hyperparameter configuration
- **Inference:** CCS prediction on unseen molecules

### Command-Line Tools

#### Parameter Optimization
```bash
python experiment_runner.py \
  --input data/smiles.csv \
  --output results/ \
  --resolutions 32 64 128 \
  --rotations 3 5 10 \
  --replicates 5
```

#### Batch Inference
```bash
python batch_inference.py \
  --model models/optimal_3dcnn.h5 \
  --projections data/projections/ \
  --output predictions.csv
```

---

## Experimental Workflow

### Experiment 1: Resolution & Rotation Optimization
- Resolution range: **8×8 to 192×192**
- Rotations per axis: **1–10**
- 40 replicates per configuration
- Training: 250 epochs, batch size 8, Adam (lr=1e-4)
- Metric: **Mean RPE** per adduct class

### Experiment 2: 4-Fold Cross-Validation
- Optimal parameters from Experiment 1
- Stratified 4-fold split by adduct
- 15 replicates with different random seeds
- Loss: **Huber loss**

---

## Model Architecture

- Input: *(N_rotations × H × W × Channels)*
- 4 × 3D convolution blocks (32 → 64 → 128 → 256 filters)
- Batch normalization + anisotropic pooling
- GlobalAveragePooling3D
- Dense layers: 128 → 64 (LeakyReLU, Dropout 0.3)
- Output: Single linear neuron (CCS regression)

Optimizer: **Adam (lr=0.0001)**  
Loss: **Huber**  
Metrics: MSE, MAE, MAPE

---

## Repository Structure

```text
Deep3DCCS/
├── 3DCNN_gui.py
├── _core.py
├── utility_modules.py
├── helper_tools.py
├── ui_interface.ui
├── datasets/
├── models/
├── Evaluations/
├── configs/
├── assets/
└── requirements.txt
```

---

## Evaluation Metrics

- **Relative Percentage Error (RPE)**
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Pearson Correlation (r)
- Coefficient of Determination (R²)
- Deming regression parameters

**Interpretation Guide:**
- RPE < 3%: Excellent
- 3–5%: Good
- 5–10%: Moderate
- >10%: Requires refinement

---

## Benchmarking

Deep3DCCS is benchmarked against:
- **CCSBASE**
- **ALLCCS**

Evaluation includes RPE distribution, paired statistical tests, computational efficiency, and external dataset validation.

---

## Output Files

- **3D Structures:** SDF + Cartesian coordinate files
- **2D Projections:** PNG stacks with JSON metadata
- **Models:** HDF5 (.h5) + checkpoints
- **Results:** CSV, JSON summaries, publication-quality plots

---

## Citation

If you use Deep3DCCS, please cite:

```bibtex
@article{deep3dccs2024,
  title={Deep3DCCS: Deep Learning Approach for CCS Prediction from Multi-Angle Projections of Molecular Geometry},
  author={Siriraj Metabolomics \& Phenomics Center},
  journal={Journal of Cheminformatics},
  year={2024},
  publisher={Springer}
}
```

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## Contact

**Siriraj Metabolomics & Phenomics Center (SiMPC)**  
Faculty of Medicine Siriraj Hospital  
Mahidol University, Bangkok 10700, Thailand

---

## Funding & Acknowledgments

Supported by:
- National Higher Education Science Research and Innovation Policy Council (NXPO)
- PMU-B (Program Management Unit for Human Resource & Institutional Development)

Computational resources provided by the **SiMPC High-Performance Computing Facility**.

---

> **Note:** Deep3DCCS is an active research project. Please check the GitHub repository for updates, report issues, and refer to documentation for advanced usage.

