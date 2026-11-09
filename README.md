# MNIST Handwritten Digit Classification — Decision Trees, AdaBoost & Stacking

**Python · scikit-learn · TensorFlow/Keras · Ensemble Learning · Multiclass Classification**

## Overview

This project explores supervised machine learning for recognizing handwritten digits (0–9) from the **MNIST** dataset. The focus is on comparing individual classifiers with ensemble approaches, especially **AdaBoost** and **stacking**, and investigating how model selection and hyperparameters influence classification performance.

The analyses were developed as Jupyter notebooks for academic learning and experimentation. In addition to final models, the notebooks intentionally retain intermediate experiments and their outputs.

## Dataset

MNIST contains **70,000 grayscale images** of handwritten digits, each measuring **28 × 28 pixels**:

- **60,000** training images
- **10,000** test images
- **10 classes** (digits 0–9)

The notebooks retrieve the dataset using `tensorflow.keras.datasets.mnist.load_data()`, normalize pixel values from 0–255 to 0–1, and reshape images into 784-dimensional feature vectors for scikit-learn models. The dataset itself is **not included** in this repository.

## Project structure

```text
mnist-ensemble-learning/
├── README.md
├── MNIST_DecisionTree.ipynb
├── Boosted_DecisionTree.ipynb
└── Ensemble_Learning_Stacking_MNIST.ipynb
```

| Notebook | Main focus |
| --- | --- |
| `MNIST_DecisionTree.ipynb` | Decision tree training, cross-validation, selection of tree depth and splitting criterion, and exploratory ensemble experiments |
| `Boosted_DecisionTree.ipynb` | AdaBoost with decision-tree base estimators; investigation of tree depth, number of estimators, and learning rate |
| `Ensemble_Learning_Stacking_MNIST.ipynb` | Comparison of base classifiers, stacking ensembles, meta-classifiers, hyperparameter experiments, and confusion-matrix analysis |

## Methods

1. **Preprocessing:** Pixel normalization and flattening of 28 × 28 images.
2. **Baseline classifiers:** Decision tree, k-nearest neighbors, and logistic regression.
3. **Model tuning:** Stratified cross-validation and parameter comparisons for tree-based classifiers and the stacking meta-classifier.
4. **Boosting:** AdaBoost using decision trees as base learners.
5. **Stacking:** Combining predictions from heterogeneous classifiers, including decision trees, kNN, random forests, and logistic regression. Variants with an AdaBoost base learner are also evaluated.
6. **Evaluation:** Classification accuracy and confusion matrices, with attention to frequently confused digits.

## Selected results

The following values are reported in the saved outputs of the notebooks. **They come from different experiments and should not be interpreted as a fully controlled benchmark under identical tuning conditions.**

| Experiment / model | Reported test accuracy | Source notebook |
| --- | ---: | --- |
| Tuned decision tree | **88.55%** | `MNIST_DecisionTree.ipynb` |
| k-nearest neighbors | **97.05%** | `Ensemble_Learning_Stacking_MNIST.ipynb` |
| Logistic regression | **92.63%** | `Ensemble_Learning_Stacking_MNIST.ipynb` |
| AdaBoost with decision-tree estimators | **97.88%** | `Boosted_DecisionTree.ipynb` |
| Tuned stacking ensemble | **97.95%** | `Ensemble_Learning_Stacking_MNIST.ipynb` |
| Stacking variant with AdaBoost | **97.89%** | `Ensemble_Learning_Stacking_MNIST.ipynb` |

### Key observations

- A single decision tree performed substantially worse than the stronger ensemble approaches.
- Boosting improved predictive performance over the standalone tree.
- The tuned stacking model reached **97.95% test accuracy** in the main stacking notebook.
- Replacing the individual decision tree with an AdaBoost-based learner **did not improve** that stacking result in the recorded experiment.
- The confusion matrices reveal remaining errors between visually similar digits (for example, 3/5 and 4/9).

## How to run

1. Install Python and JupyterLab or Jupyter Notebook.
2. Install the main dependencies:

   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow graphviz jupyterlab
   ```

3. Open the notebooks from the repository root and run their cells in order:

   ```bash
   jupyter lab
   ```

**Notes:** The notebooks download MNIST automatically. Some cross-validation and ensemble experiments are computationally expensive. The notebooks use `graphviz` for tree visualization; displaying rendered graphs may also require installing the Graphviz system package. Exact dependency versions are not pinned, so behavior may differ across library versions.

## Limitations and possible improvements

- Some notebook sections reflect exploratory parameter searches rather than a single automated end-to-end pipeline.
- Several candidate configurations are compared using test-set accuracy. For a more rigorous study, all model and hyperparameter selection should be completed on training/validation data before a final one-time test evaluation.
- Runtime and predictive performance depend on the computing environment.
- MNIST is a well-established educational benchmark; results should not be taken as evidence of production-ready image recognition.

## Skills demonstrated

`Python` · `Jupyter` · `scikit-learn` · `Multiclass classification` · `Cross-validation` · `Hyperparameter tuning` · `Decision trees` · `AdaBoost` · `Random forests` · `kNN` · `Logistic regression` · `Stacking` · `Model evaluation`

## Dataset reference

LeCun, Y., Cortes, C., & Burges, C. J. C. **The MNIST Database of Handwritten Digits.** https://yann.lecun.com/exdb/mnist/

---

*Academic machine-learning project documenting both experimental workflow and selected results.*
