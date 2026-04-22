# Supervised Feature Selection Guided by Granular-Ball Computing: A Two-Stage Geometry-Driven Framework

This repository contains the official implementation of **SFS-GBC**, a two-stage geometry-driven framework for supervised feature selection.

## Overview
Unlike traditional statistical feature selection methods, SFS-GBC leverages the geometric distribution of data granules (granular balls) to evaluate feature importance. It operates in two stages:
1. **Stage I (Coarse-grained)**: Rapidly ranks features using **Granularity Consistency (GC)**.
2. **Stage II (Fine-grained)**: Refines the feature subset via forward search guided by **Intra-class Compactness (ICC)** and **Inter-class Discrimination (ICD)**.



## Requirements
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- SciPy

## Usage
Prepare your dataset in CSV format where the last column is the target class.

```python
from SFS_GBC_Release import SFS_GBC
import pandas as pd

# Load dataset
df = pd.read_csv('your_data.csv')

# Initialize and run
selector = SFS_GBC(data=df, lam='')
selected_features = selector.run()

print(f"Optimal feature indices: {selected_features}")
