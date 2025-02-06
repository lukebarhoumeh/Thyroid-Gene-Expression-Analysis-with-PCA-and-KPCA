# Thyroid Cancer Gene-Expression Analysis

This repository demonstrates a **dimensionality reduction** and **classification** workflow on Thyroid cancer gene-expression data. We explore **PCA** (Principal Component Analysis), **Kernel PCA** (KPCA) with RBF, Polynomial, and Linear kernels, and compare them to a simple top-10-feature subset (selected by covariance analysis). We then train different classifiers (e.g., KNN, Naive Bayes, etc.) to evaluate **accuracy**, **precision**, and **recall** under various dimensionality‐reduction scenarios.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Dimensionality Reduction Techniques](#dimensionality-reduction-techniques)
   - [PCA (From scikit-learn)](#pca-from-scikit-learn)
   - [Kernel PCA (KPCA) From Scratch](#kernel-pca-kpca-from-scratch)
   - [Top 10 Features (Covariance Analysis)](#top-10-features-covariance-analysis)
4. [Classifiers](#classifiers)
5. [Usage](#usage)
6. [Results and Plots](#results-and-plots)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Project Overview

1. **Goal**: Reduce extremely high-dimensional Thyroid cancer gene-expression data to fewer components and compare classification performance using different approaches.
2. **Key Steps**:
   - Load and split the Thyroid dataset (CSV) into train/test.
   - Implement or use:
     - **PCA** (scikit-learn)
     - **KPCA** (RBF, Polynomial, Linear) from scratch
     - A **top-10** subset of features from covariance matrix analysis
     - **No dimensionality reduction** as a baseline
   - Train classifiers (e.g., KNN, Naive Bayes) on each reduced dataset.
   - Evaluate **accuracy**, **precision**, and **recall**, and plot “accuracy vs. number of components.”

---

## Data Description

- **Dataset**: `Thyloid.csv` (a CSV file containing gene-expression measurements).
- **Rows**: Each row corresponds to one patient/sample.
- **Columns**: 
  - **Features**: numerous gene-expression levels (e.g., 1881 columns).
  - **Label**: The last column indicates the class/label (e.g., Cancer type, normal vs. tumor, etc.).

---

## Dimensionality Reduction Techniques

### PCA (From scikit-learn)

1. Use `PCA(n_components=k)` from `sklearn.decomposition`.
2. Fits on the **train** set and transforms **test**.
3. Retains up to 95% variance or a fixed number `k` for comparison.

### Kernel PCA (KPCA) From Scratch

We implemented KPCA by:
1. **Building** the kernel matrix `K` (RBF, Polynomial, or Linear).
2. **Centering** `K` in feature space.
3. **Eigen-decomposing** the centered matrix to get top components.
4. **Transforming** data (both train and test).

#### RBF Kernel
\[
K_{ij} = \exp(-\gamma \|x_i - x_j\|^2)
\]

#### Polynomial Kernel
\[
K_{ij} = (\alpha \, x_i^\top x_j + c)^{d}
\]

#### Linear Kernel
\[
K_{ij} = x_i^\top x_j
\]

### Top 10 Features (Covariance Analysis)

- Compute the **covariance matrix** of the full dataset.
- Select the **10 features** with the largest variances on the diagonal.
- Use only those 10 columns (features) for training/testing classifiers.

---

## Classifiers

- **KNN** (scikit-learn)
- **Naive Bayes** (scikit-learn)
- (Optional) **Minimum Distance** or **Bayes** from scratch
- Each classifier is trained separately on each dimensionality‐reduced dataset.

---

## Usage

1. **Clone** this repository.
2. **Place** your `Thyloid.csv` file in the same directory (or adjust paths accordingly).
3. **Run** the notebook or script (e.g., `python your_script.py`).
   - The code loads `Thyloid.csv`.
   - Splits it into `train` / `test`.
   - Performs PCA/KPCA.
   - Trains KNN/Naive Bayes on each reduced dataset.
   - Prints classification metrics and plots.

4. **Plot**: You’ll see “accuracy vs. number of components” plots and final accuracy, precision, and recall for each approach.

---

## Results and Plots

- **Best number of components**: For each method (PCA, RBF-KPCA, etc.), we find the “best k” that maximizes accuracy.
- **Accuracy vs. k**: Plots show how accuracy varies with number of components for different kernels.
- **Precision & Recall**: Summaries of precision and recall highlight whether the model predicts specific classes effectively.
- **Top-10** approach**: Sometimes competes well with KPCA or PCA, but typically less flexible or sophisticated than kernel methods.

---

## Conclusion

- **PCA** provides a strong baseline for linear dimensionality reduction, often capturing enough variance with fewer components.
- **KPCA** can capture nonlinear relationships—sometimes boosting classification accuracy beyond linear PCA.
- **Top 10 features** is a quick, interpretable approach, but lacks the holistic variance capture of PCA or KPCA.
- Overall, **kernel** choice (RBF, Polynomial, Linear) and **number of components** significantly affect results.
- Implementing these methods from scratch deepened our understanding of kernel construction, centering, and eigen-decomposition. Using scikit-learn’s built-ins is more convenient and robust for larger-scale or production scenarios.

---

## References

1. [scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  
2. [scikit-learn KernelPCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)  
3. [Numpy Documentation](https://numpy.org/doc/stable/)  
4. [Matplotlib Documentation](https://matplotlib.org/stable/index.html)  

---
