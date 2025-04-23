# Exceptional Model Mining for Survival Analysis
_Integrating a local pattern mining framework into an Early Warning Score system_

This repository contains the code and supplementary materials for my graduation thesis, in which I extend the [Exceptional Model Mining (EMM) framework](https://link.springer.com/article/10.1007/s10618-015-0403-4) to several survival analysis models to reduce false alerts in the Early Warning Score (EWS) system at Catharina Ziekenhuis. The EWS detects patient deterioration and raises alerts. By identifying “exceptional” subgroups, we aim to help healthcare professionals adjust the existing fixed physiological thresholds, which are applied to the overall population or only a few broad subgroups.

**Thesis:** [Exceptional Survival Analysis](Link) (The link will be updated when it is published)

---

## Requirements

- Python 3.8+ 
- See [requirements.txt](requirements.txt) for exact versions.

---

## Data

- All patient data belong to Catharina Ziekenhuis and are excluded here.
- A user can construct hypothetical datasets to run this code whose columns include patient ID and admission ID (both in UUID format), several time-varying variables (e.g., heart rate), static variables (e.g., sex), plus target variables for the time to first deterioration event and an indicator of whether the event occurred.


## Installation

1. **Clone** the repo  
   ```bash
   git clone https://github.com/dyk00/Exceptional_Survival_Analysis.git
   ```
2. **Install** dependencies  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

> _Note: All Jupyter notebooks have been cleared of outputs to avoid exposing any patient‐identifiable information_

### 1. Survival Analysis

```bash
cd survival_analysis
python main.py
```

This script will:

- Fit KM, LR. 
- Fit each survival model (Cox, CoxNet, ERSF, RSF, Weibull) using seeds 1, 10, and 42.  
- Output estimated survival probabilities and performance metrics.
- Generate corresponding plots.

### 2. Exceptional Model Mining

```bash
python main.py
```

- Reads in each model output
- Identifies exceptional subgroups where survival curves are different

For finding matching subgroups across different models and datasets, use:

```bash
python supplementary_material/emm_result/matching.py
```

---

## Supplementary Material

### `emm_result/`
- Contains results from the EMM framework:
  - AAM model output  
  - Survival model outputs for Cox, CoxNet, ERSF, RSF, and Weibull (random seeds: 1, 10, 42)  
- Includes the matching(overlapping) subgroups

### `report/`
- Contains the basic visual analytics produced by the `y_data_profiling` package

---

## License

This project is released under the [MIT License](LICENSE).


## Copyright

**`aam_modified.py`** reproduces the AAM model from [P. Kipnis et al](https://pubmed.ncbi.nlm.nih.gov/27658885/).  
Author: Tom Bakkes  
Modified for EMM by Dayeong Kim
