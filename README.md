<div align="center">
  <img src="figures/logo.png" alt="Orion-MSP Logo" width="700"/>
</div>

<div align="center">
  <a href="https://www.lexsi.ai/">
    <img src="https://img.shields.io/badge/Lexsi-Homepage-FF6B6B?style=for-the-badge" alt="Homepage"/>
<a href="https://huggingface.co/Lexsi/Orion-MSP">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Lexsi AI-FFD21E?style=for-the-badge" alt="Hugging Face"/>
  </a>
  </a>
  <a href="https://discord.gg/dSB62Q7A">
    <img src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"/>
  </a>
</div>


# Orion-MSP: Multi-Scale Sparse Attention for Tabular In-Context Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[Orion-MSP](https://arxiv.org/abs/2511.02818)** is a tabular foundation model that combines **multi-scale sparse attention** with **Perceiver-style memory** for efficient in-context learning on tabular data. The model processes features at multiple resolutions simultaneously, capturing both local feature interactions and global dataset-level patterns through hierarchical attention mechanisms.

## Approach and Architecture

### Key Innovations

Orion-MSP introduces three key architectural innovations:

1. **Multi-Scale Sparse Attention**: Processes features at multiple granularities (scales 1, 4, 16) using windowed, global, and random attention patterns. This hierarchical approach captures feature interactions at different granularities—from individual features to feature groups—reducing computational complexity to near-linear.

2. **Hierarchical Feature Understanding**: Combines representations across multiple scales to balance local precision and global context, enabling robust performance on datasets with varying feature counts and complexity.

3. **Perceiver-Style Cross-Component Memory**: Maintains a compressed memory representation across dataset samples using a latent memory system. This enables efficient bidirectional information flow between model components while preserving in-context learning safety constraints.

### Architecture Components

The model consists of four main components:

1. **Column-wise Embedding**: Distribution-aware feature embeddings using Induced Set Attention Blocks (ISAB). This component is adopted from TabICL and transforms raw tabular cells into high-dimensional embeddings while capturing statistical regularities within each column.

2. **Multi-Scale Row Interaction**: Hierarchical sparse attention module that processes features at multiple scales:
   - **Scale 1**: Individual feature interactions with maximal detail
   - **Scale 4**: Feature groups capturing local relationships
   - **Scale 16**: Coarse-grained patterns for global context
   - Each scale uses block-sparse attention with sliding windows, global tokens, and optional random links

3. **Cross-Component Memory**: Perceiver-style latent memory that compresses dataset information:
   - Training rows write context to latent memory
   - All rows read from memory to obtain enhanced representations
   - Maintains ICL safety by preventing information leakage between train and test data

4. **Dataset-wise ICL Predictor**: Enhanced predictor that leverages enriched representations for few-shot tabular classification using split self-attention with label injection.

## Installation

### Prerequisites

- Python 3.9-3.12
- PyTorch 2.2+ (with CUDA support recommended)
- CUDA-capable GPU (recommended for training)

### From the source

#### Option 1: From the local clone

```bash
cd orion-msp
pip install -e .
```

#### Option 2: From the Git Remote
```bash
pip install git+https://github.com/Lexsi-Labs/Orion-MSP.git
```

### Dependencies
The package will automatically install required dependencies:
- `torch>=2.2,<3`
- `scikit-learn>=1.7,<2.0`
- `numpy`, `scipy`, `joblib`
- `xgboost`
- `transformers`
- `einops>=0.7`
- `huggingface-hub`
- `wandb` (for training)

## Usage

### Basic Usage

Orion-MSP provides a scikit-learn compatible interface for easy integration:

```python
from orion_msp.sklearn import OrionMSPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the classifier
clf = OrionMSPClassifier()

# Fit the model (prepares data transformations)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

### Advanced Configuration

#### Model Hyperparameters

The `OrionMSPClassifier` supports extensive configuration options:

```python
from orion_msp.sklearn import OrionMSPClassifier

clf = OrionMSPClassifier(
    # Ensemble parameters
    n_estimators=32,                    # Number of ensemble members (default: 32)
    norm_methods=["none", "power"],      # Normalization methods: "none", "power", "quantile", "quantile_rtdl", "robust"
    feat_shuffle_method="shift",        # Feature permutation: "none", "shift", "random", "latin"
    class_shift=True,                   # Apply cyclic shifts to class labels (default: True)
    outlier_threshold=4.0,              # Z-score threshold for outlier clipping (default: 4.0)
    
    # Prediction parameters
    softmax_temperature=0.9,             # Temperature for softmax (default: 0.9)
    average_logits=True,                 # Average logits vs probabilities (default: True)
    use_hierarchical=True,              # Enable hierarchical classification for many classes (default: True)
    
    # Inference parameters
    use_amp=True,                       # Use automatic mixed precision (default: True)
    batch_size=4,                       # Batch size for ensemble inference (default: 8)
    device="cuda",                      # Device: "cuda", "cpu", or None for auto
    
    # Model checkpoint
    model_path=None,                    # Path to checkpoint (None = auto-download from Hugging Face)
    allow_auto_download=True,           # Auto-download checkpoint if not found (default: True)
    checkpoint_version="Orion-MSP-v1.0.ckpt",  # Checkpoint version, new versions will be available soon
    
    # Other parameters
    random_state=42,                    # Random seed for reproducibility
    n_jobs=None,                        # Number of threads for CPU inference
    verbose=False,                      # Print detailed information
)
```

#### Model Architecture Parameters

When training from scratch or loading custom checkpoints, you can specify model architecture:

```python
from orion_msp.model.orion_msp import OrionMSP

model = OrionMSP(
    # Basic parameters
    max_classes=10,                     # Maximum number of classes (default: 10)
    embed_dim=128,                      # Embedding dimension (default: 128)
    
    # Column embedding
    col_num_blocks=3,                   # Number of Set Transformer blocks (default: 3)
    col_nhead=4,                        # Number of attention heads (default: 4)
    col_num_inds=128,                   # Number of inducing points (default: 128)
    
    # Multi-scale sparse row interaction
    row_num_blocks=6,                   # Number of Transformer blocks per scale (default: 6)
    row_nhead=8,                        # Number of attention heads (default: 8)
    row_num_cls=4,                      # Number of CLS tokens (default: 4)
    row_rope_base=100000,               # RoPE base frequency (default: 100000)
    row_num_global=4,                   # Number of global tokens (default: 4)
    row_scales=(1, 4, 16),              # Multi-scale resolutions (default: (1, 4, 16))
    row_window=8,                       # Sliding window size (default: 8)
    row_num_random=2,                   # Number of random attention links (default: 2)
    row_group_mode="pma",               # Feature grouping: "contiguous" or "pma" (default: "pma")
    
    # Perceiver memory
    perc_num_latents=32,                # Number of latent vectors (default: 32, 0 to disable)
    perc_layers=2,                      # Number of Perceiver layers (default: 2)
    
    # ICL predictor
    icl_num_blocks=12,                  # Number of ICL blocks (default: 12)
    icl_nhead=4,                        # Number of attention heads (default: 4)
    
    # Shared parameters
    ff_factor=2,                        # Feedforward dimension multiplier (default: 2)
    dropout=0.0,                        # Dropout probability (default: 0.0)
    activation="gelu",                  # Activation function (default: "gelu")
    norm_first=True,                    # Pre-norm architecture (default: True)
)
```

## Preprocessing

Orion-MSP includes automatic preprocessing that handles:

1. **Categorical Encoding**: Automatically encodes categorical features using ordinal encoding
2. **Missing Value Imputation**: Handles missing values using median imputation for numerical features
3. **Feature Normalization**: Supports multiple normalization methods:
   - `"none"`: No normalization
   - `"power"`: Yeo-Johnson power transform
   - `"quantile"`: Quantile transformation to normal distribution
   - `"quantile_rtdl"`: RTDL-style quantile transform
   - `"robust"`: Robust scaling using median and quantiles
4. **Outlier Handling**: Clips outliers beyond a specified Z-score threshold (default: 4.0)
5. **Feature Permutation**: Applies systematic feature shuffling for ensemble diversity:
   - `"none"`: Original feature order
   - `"shift"`: Circular shifting
   - `"random"`: Random permutation
   - `"latin"`: Latin square patterns (recommended)

The preprocessing is automatically applied during `fit()` and `predict()`, so no manual preprocessing is required.

## Performance

### Benchmark Results

Orion-MSP is the most consistent top performer across all benchmarks, achieving the best overall rank.

<div align="center">

<table>
<caption><strong>Performance comparison across three benchmark suites—TALENT, OpenML-CC18, and TabZilla.</strong> Rank is accuracy-based ranking (lower is better). Metrics: ACC = Accuracy, F1 = Weighted F1.</caption>
<thead>
<tr>
<th rowspan="2" style="text-align: left; padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">Models</th>
<th colspan="1" style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">All</th>
<th colspan="3" style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">TALENT</th>
<th colspan="3" style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">OpenML-CC18</th>
<th colspan="3" style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">TabZilla</th>
</tr>
<tr>
<th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">Rank</th>
<th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">Rank</th><th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">ACC</th><th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">F1</th>
<th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">Rank</th><th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">ACC</th><th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">F1</th>
<th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">Rank</th><th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">ACC</th><th style="padding: 8px; border: 1px solid #ddd; background-color: #f9f9f9;">F1</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;">XGBoost</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.70</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.02</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8403</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8360</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">5.89</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8558</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8537</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.07</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8612</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8326</td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;">CatBoost</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.43</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">5.57</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8336</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8259</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.25</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8588</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8520</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">7.13</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8579</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8384</td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;">Random Forest</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">7.38</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.15</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8285</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8209</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.36</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8547</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8497</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">8.42</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8358</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8399</td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;">LightGBM</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.78</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.11</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8331</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8245</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.18</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8581</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8493</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">5.25</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8618</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8211</td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;">TabICL</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">4.96</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">4.09</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">0.8471</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">0.8379</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">4.69</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8667</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8623</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">5.89</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8734</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8698</td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;">OrionBiX</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">5.37</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">4.59</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8346</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8260</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">4.98</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8653</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8596</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">4.89</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8728</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8628</td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;"><strong>OrionMSP</strong></td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;"><strong>3.58</strong></td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; font-weight: bold; text-decoration: underline;"><strong>3.26</strong></td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8461</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8360</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; font-weight: bold; text-decoration: underline;"><strong>4.12</strong></td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; font-weight: bold; text-decoration: underline;"><strong>0.8722</strong></td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; font-weight: bold; text-decoration: underline;"><strong>0.8676</strong></td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; font-weight: bold; text-decoration: underline;"><strong>3.84</strong></td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; font-weight: bold; text-decoration: underline;"><strong>0.8821</strong></td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; font-weight: bold; text-decoration: underline;"><strong>0.8786</strong></td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;"><u>TabPFN</u></td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">4.61</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">3.72</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; font-weight: bold; text-decoration: underline;"><strong>0.8514</strong></td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; font-weight: bold; text-decoration: underline;"><strong>0.8412</strong></td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">4.76</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">0.8714</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">0.8663</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">4.86</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8752</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8716</td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;">Mitra</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">11.77</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">10.38</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.3921</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.2868</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">10.52</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.3614</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.2522</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">11.21</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.3152</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.1830</td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;">ContextTab</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">9.70</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">9.84</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.5474</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.4596</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">6.28</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8639</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8581</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">7.13</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8389</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8334</td>
</tr>
<tr>
<td style="text-align: left; padding: 6px 8px; border: 1px solid #ddd;">TabDPT</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">5.42</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">5.19</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8408</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8318</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">4.64</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8672</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd;">0.8625</td>
<td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">3.94</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">0.8814</td><td style="text-align: center; padding: 6px 8px; border: 1px solid #ddd; text-decoration: underline;">0.8775</td>
</tr>
</tbody>
</table>

</div>

### Performance Plots

<div align="center">
  <img src="figures/accuracy_ranking_talent.png" alt="Accuracy Ranking TALENT" width="700"/>
</div>

<div align="center">
  <img src="figures/accuracy_ranking_tabzilla.png" alt="Accuracy Ranking TabZilla" width="700"/>
</div>

<div align="center">
  <img src="figures/accuracy_ranking_openml-cc18.png" alt="Accuracy Ranking OPENML-CC18" width="700"/>
</div>

<div align="center">
  <table>
    <tr>
      <td style="padding: 5px;"><img src="figures/relative_acc_improvement_over_talent.png" alt="Relative Improvement over XGBoost on TALENT" width="600"/></td>
      <td style="padding: 5px;"><img src="figures/relative_acc_improvement_over_tabzilla.png" alt="Relative Improvement over XGBoost on TabZilla" width="600"/></td>
    </tr>
  </table>
</div>

<div align="center">
  <table>
    <tr>
      <td style="padding: 3px;"><img src="figures/relative_acc_improvement_over_talent.png" alt="Relative Improvement over XGBoost on TALENT" width="500"/></td>
    </tr>
  </table>
</div>

## Citation

If you use Orion-MSP in your research, please cite our [paper](https://arxiv.org/abs/2511.02818):

```bibtex
@article{bouadi25orionmsp,
  title={Orion-MSP: Multi-Scale Sparse Attention for Tabular In-Context Learning},
  author={Mohamed Bouadi and Pratinav Seth and Aditya Tanna and Vinay Kumar Sankarapu},
  year={2025}
  eprint={2511.02818},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2511.02818}, 
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions, issues, or contributions, please:
- Open an issue on [GitHub](https://github.com/Lexsi-Labs/Orion-MSP/issues)
- Join our [Discord](https://discord.gg/dSB62Q7A) community

