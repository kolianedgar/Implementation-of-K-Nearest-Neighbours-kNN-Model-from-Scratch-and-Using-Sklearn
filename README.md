# k-Nearest Neighbours (kNN): Custom & Package Implementations with Statistical Testing

This repository contains a comprehensive implementation and evaluation of the **k-Nearest Neighbours (kNN)** algorithm. It combines:

* A **custom, from-scratch implementation** of kNN
* A **package-based implementation** using standard Python libraries
* **Statistical testing** to rigorously compare performance across models and configurations

This work was developed as part of the coursework for the university course **LZSCC.461: Programming for Data Scientists**, with a focus on correctness, reproducibility, and empirical evaluation.

---

## 🔍 Project Overview

### Custom kNN Implementation

The custom kNN implementation is written without relying on external machine learning libraries and includes:

* Configurable number of neighbours (`k`)
* Multiple distance metrics (e.g. Euclidean, Manhattan)
* Support for multiple datasets
* Classification, accuracy, and evaluation metrics
* Utilities for reporting and memory handling

This implementation demonstrates a clear understanding of the internal mechanics of the kNN algorithm.

---

### Package-Based kNN

A package-based implementation (e.g. using `scikit-learn`) is included to:

* Provide a benchmark against the custom implementation
* Validate correctness and consistency of results
* Compare performance under identical experimental conditions

---

### Statistical Testing

To ensure meaningful comparisons, the project applies statistical testing to experimental results, including:

* Multiple experimental runs
* Hypothesis testing for performance differences
* Comparisons between custom and package-based implementations
* Analysis across different datasets and values of `k`

This ensures conclusions are statistically justified rather than anecdotal.

---

## 🧪 Datasets

The following datasets are used throughout the experiments:

* Iris
* Wine
* Digits
* Breast Cancer
* Zoo
* Mushroom (Agaricus-Lepiota)

These datasets vary in size, dimensionality, and class distribution to test robustness across different problem settings.

---

## ▶️ Running the Code

### Install the project

Dependencies are managed via `pyproject.toml`. Install the project with:

```bash
pip install .
```

### Run tests

```bash
pytest
```

Each test evaluates the kNN implementation on a specific dataset and configuration.

---

## 📊 Results

Results include:

* Classification accuracy and evaluation metrics
* Aggregated experimental outputs
* Statistical test summaries

Detailed outputs can be found in:

```
Own Implementation/test-output-own.txt
```

---

## 📌 Learning Outcomes

* Implementing kNN from first principles
* Comparing machine learning implementations fairly
* Applying statistical testing to ML experiments
* Designing reproducible experimental pipelines

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
