# Stability and Interpretability of Penalized Logistic Regression Models for Breast Cancer Risk Prediction

This repository contains the data, R scripts, and supporting materials used to reproduce the analyses presented in the manuscript:

> **Stability and Interpretability of Penalized Logistic Regression Models for Breast Cancer Risk Prediction**

The repository has been made publicly available to promote transparency, reproducibility, and open scientific research.

---

## Repository Structure

```
├── data/        # Breast Cancer Wisconsin Diagnostic Dataset
├── R/           # R scripts for data preprocessing, model development, evaluation, and bootstrap stability selection
├── results/     # Figures and graphical outputs generated from the analyses
└── README.md
```

---

## Data

The study uses the **Breast Cancer Wisconsin Diagnostic Dataset**, which is publicly available from the UCI Machine Learning Repository.
https://www.kaggle.com/code/abdulrhmansalama/breast-cancer-wisconsin-dataset

---

## Software Requirements

Analyses were conducted using:

- **R version 4.5.0**

### Required R Packages

- glmnet
- caret
- pROC
- rms
- ggplot2
- dplyr
- tidyr

Install the required packages using:

```r
install.packages(c(
  "glmnet",
  "caret",
  "pROC",
  "rms",
  "ggplot2",
  "dplyr",
  "tidyr"
))
```

---

## Reproducibility

The R scripts included in this repository reproduce the analyses presented in the manuscript, including:

- Data preprocessing
- Penalized logistic regression models (Ridge, LASSO, and Elastic Net)
- Model performance evaluation
- Calibration assessment
- Bootstrap stability selection
- Generation of figures and results

---

## Citation

If you use the data or code from this repository, please cite the associated manuscript:

> Okyere, F., et al. *Stability and Interpretability of Penalized Logistic Regression Models for Breast Cancer Risk Prediction.* (Under review)

---

## License

This repository is made available for academic and research purposes. Please appropriately acknowledge the original authors when using or adapting the materials.

---

## Contact

**Francis Okyere**
Department of Statistics  
Florida State University


Email: kokyere.gh@gmail.com
