# DeepSDF – Dataset Analysis and Distribution Correction


# 📌 Overview

This project investigates **3D shape reconstruction using DeepSDF**, with a strong focus on **data quality and distribution**.

The main contribution is the identification of a **critical imbalance in the Signed Distance Function (SDF) dataset**, and the design of a pipeline to **analyze, correct, and validate** its impact on model performance.

---

## 🧠 Background

DeepSDF models a shape as a continuous function:

- SDF(x) < 0 → point is inside the object  
- SDF(x) > 0 → point is outside the object  
- SDF(x) = 0 → surface  

For the model to learn correctly, the dataset must include a **balanced representation of interior, surface, and exterior points**.

---

## ⚠️ Problem Statement

Initial experiments were conducted on a large-scale dataset (~1M points) representing an airport scene.

Despite multiple training attempts and hyperparameter tuning, the model produced:

- No coherent geometry  
- Highly noisy reconstructions  
- Lack of structural consistency  

---

## 🔍 Root Cause Analysis

A detailed analysis of the dataset revealed a **critical distribution issue**:

### Initial Dataset (`LFPG_sdf_train.parquet`)

- ~91% surface points  
- ~9% interior points  
- **0% exterior points**  
- SDF max ≈ 0 → no positive values  

➡️ The model never observes the outside region.

### Consequence

DeepSDF learns a **signed function**, but:

- Without exterior samples → positive SDF cannot be learned  
- Surface (SDF = 0) becomes ill-defined  
- Reconstruction fails completely  

---

## 🔧 Proposed Solution

A new **balanced dataset** was generated:

### Balanced Dataset (`LFPG_sdf_train_balanced.parquet`)

- ~79% surface points  
- ~10% interior points  
- ~10% exterior points  
- SDF mean ≈ 0 (symmetric distribution)

### Key Idea

Artificially generate **exterior points** using controlled noise sampling.

➡️ This restores a **valid SDF signal** for learning.

---

## 📊 Dataset Comparison

| Metric        | Initial Dataset | Balanced Dataset |
|--------------|---------------|------------------|
| Surface      | 91.37%        | 78.94%           |
| Interior     | 8.63%         | 10.53%           |
| Exterior     | 0.00%         | 10.54%           |
| SDF Mean     | -0.0107       | ~0               |
| SDF Range    | [-0.57, ~0]   | [-0.108, 0.102]  |

---

## 🧪 Pipeline

The full pipeline includes:

1. **Dataset inspection**
   - Visualization with threshold ε
   - SDF distribution analysis

2. **Problem identification**
   - Missing exterior samples
   - Strong surface bias

3. **Data correction**
   - Exterior point generation
   - Noise-controlled sampling
   - Distribution balancing

4. **Spatial decomposition**
   - Scene split into tiles (4×4 grid)
   - Local training

5. **Model training**
   - DeepSDF with Fourier features
   - SDF clamping for stability

6. **Evaluation**
   - Mesh reconstruction (Marching Cubes)
   - Visual comparison

---
## 📁 Project Structure

├── data/
│ ├── LFPG_sdf_train.parquet
│ ├── LFPG_sdf_train_balanced.parquet
│
├── tools/
│ ├── analyze_sdf_distribution.py
│ ├── clean_sdf_parquet.py
│ ├── add_exterior_points.py
│ ├── make_tiles.py
│
├── training/
│ ├── train.py
│ ├── airport_dataset.py
│
├── evaluation/
│ ├── grid_eval.py
│
├── outputs/
│
└── README.md

---

