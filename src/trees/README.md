# CART Decision Tree Implementation Journal (Julia)

## Project Overview

**Project:** CART (Classification and Regression Trees) from First Principles

**Language:** Julia

**Objective:** Build a complete CART classification implementation without relying on machine learning libraries in order to understand the underlying algorithms and mathematics.

This project is intended as both a software engineering exercise and a statistical learning exercise. Each component is implemented incrementally, tested independently, and documented before integration into the complete tree-building algorithm.

The implementation has been tested on both small synthetic datasets and the Iris dataset.

---

# Mathematical Background

The Classification and Regression Tree (CART) algorithm builds a decision tree by recursively partitioning the feature space into increasingly homogeneous regions. At each node, the algorithm searches for the feature and threshold that produces the best split according to an impurity measure.

Unlike many statistical methods that optimize a global objective directly, CART follows a **greedy optimization strategy**. Each split is chosen solely to maximize the immediate reduction in impurity at the current node. Although this does not guarantee the globally optimal tree, it produces effective models while remaining computationally tractable.

---

## Node Impurity

For classification problems, CART measures the impurity of a node using **Gini Impurity**.

Suppose a node contains observations from $K$ classes. Let

$$
p_k =
\frac{\text{Number of observations in class }k}
{\text{Total observations in the node}}
$$

be the proportion of observations belonging to class $k$.

The Gini impurity is

$$
G = 1 - \sum_{k=1}^{K} p_k^2
$$

Properties:

- $G = 0$ for a pure node.
- Larger values indicate greater class mixing.
- Maximum impurity occurs when classes are equally represented.

---

## Evaluating Candidate Splits

A candidate split partitions a parent node into a left child and a right child.

If

- $n_L$ = number of observations in the left node
- $n_R$ = number of observations in the right node
- $n = n_L + n_R$

then the impurity of the split is the weighted average

$$
G_{\text{split}}
=
\frac{n_L}{n}G_L
+
\frac{n_R}{n}G_R
$$

where

- $G_L$ is the Gini impurity of the left child.
- $G_R$ is the Gini impurity of the right child.

The optimal split is the one that minimizes

$$
G_{\text{split}}
$$

across every candidate threshold and every feature.

---

## Candidate Thresholds

For a numeric feature, the feature values are sorted and candidate thresholds are generated between adjacent unique values.

For example:

```text
Feature values:

1
2
5
7
9
```

produce candidate thresholds:

```text
1.5
3.5
6.0
8.0
```

Each threshold defines a possible binary partition:

```text
x <= threshold  -> left child
x > threshold   -> right child
```

CART evaluates the weighted Gini impurity produced by each candidate threshold.

---

## Greedy Split Selection

For each feature:

1. Generate candidate thresholds.
2. Partition the labels at each threshold.
3. Calculate weighted Gini.
4. Retain the threshold producing the lowest weighted Gini.

The process is then repeated across all features.

The feature and threshold with the lowest weighted Gini become the decision rule for the current node.

For example:

```text
PetalLength <= 2.45
```

may become a node in the learned tree.

This is the **greedy** component of CART: the best split is selected at the current node without considering the effect of possible future splits.

---

## Boolean Masks and Data Partitioning

Splits are implemented using Boolean masks.

For example:

```julia
left_mask = X[:, feature] .<= threshold
right_mask = X[:, feature] .> threshold
```

The masks are then applied to both the feature matrix and label vector:

```julia
X_left = X[left_mask, :]
y_left = y[left_mask]

X_right = X[right_mask, :]
y_right = y[right_mask]
```

Because the same mask is applied to `X` and `y`, each observation remains associated with its correct class label.

---

## Recursive Partitioning

After selecting the best split, the same procedure is applied independently to each child node.

Conceptually:

```text
build_tree(X, y)
        |
        v
    best_split()
        |
        v
   partition data
      /       \
     /         \
build_tree()  build_tree()
   left          right
```

Each recursive call receives a smaller subset of the original dataset.

The basic stopping condition occurs when all observations reaching a node have the same label:

```julia
if length(unique(y)) == 1
```

The node is then converted into a leaf.

The implementation also stops when no valid split can be found.

---

## Maximum Tree Depth

An optional `max_depth` parameter limits recursive tree growth.

The root node has

```text
depth = 0
```

and each recursive call increments the depth:

```text
depth = 0    root
depth = 1    children
depth = 2    grandchildren
...
```

If

```julia
depth >= max_depth
```

the recursion stops and the node becomes a leaf predicting the majority class of the observations reaching that node.

This provides a simple method for controlling model complexity and reducing overfitting.

---

## Prediction

Once training is complete, prediction no longer requires Gini calculations.

A new observation traverses the learned tree according to its decision rules:

```julia
if x[node.feature] <= node.threshold
    go left
else
    go right
end
```

Traversal continues until a leaf node is reached.

The class stored in the leaf is returned as the prediction.

Thus, the trained CART model can be viewed as a recursively constructed collection of `if/else` statements.

---

# Implementation Structure

The current implementation is organized into the following files:

```text
src/trees/
├── Gini.jl
├── Split.jl
├── Node.jl
├── CART.jl
├── Predict.jl
├── Metrics.jl
├── setup.jl
├── depth_experiment.jl
├── cross_validation.jl
└── README.md
```

Major responsibilities:

```text
Gini.jl
├── gini()
└── weighted_gini()

Split.jl
├── candidate_thresholds()
├── best_split_feature()
└── best_split()

Node.jl
└── TreeNode

CART.jl
├── majority_class()
├── build_tree()
└── print_tree()

Predict.jl
└── predict()

Metrics.jl
└── confusion_matrix()
```

---

# Roadmap

## Phase 1 — Foundations

### ✓ Gini Impurity

**Status:** Complete

Implemented

$$
G = 1 - \sum_{k=1}^{K}p_k^2
$$

### Learning Notes

- Gini measures node impurity.
- A pure node has $G=0$.
- Gini provides the objective used to compare candidate splits.

---

### ✓ Weighted Gini

**Status:** Complete

Implemented

$$
G_{\text{split}}
=
\frac{n_L}{n}G_L
+
\frac{n_R}{n}G_R
$$

The candidate split with the lowest weighted Gini is preferred.

---

### ✓ Candidate Threshold Generation

**Status:** Complete

Candidate thresholds are generated from midpoints between adjacent feature values.

---

### ✓ Best Split for One Feature

**Status:** Complete

For every candidate threshold:

1. Partition labels using Boolean masks.
2. Calculate weighted Gini.
3. Retain the threshold with the lowest impurity.

---

# Phase 2 — CART Algorithm

### ✓ Search Across All Features

**Status:** Complete

The algorithm evaluates every feature and selects the feature/threshold combination with the minimum weighted Gini.

---

### ✓ Node Structure

**Status:** Complete

Implemented a recursive `TreeNode` structure containing:

- prediction
- feature index
- threshold
- left child
- right child

Leaf nodes contain a prediction and no child nodes.

---

### ✓ Recursive Tree Construction

**Status:** Complete

Implemented recursive tree construction using `build_tree()`.

At each node:

1. Check stopping criteria.
2. Find the best split.
3. Partition the dataset.
4. Recursively build the left child.
5. Recursively build the right child.

---

### ◐ Stopping Criteria

**Status:** Partially Complete

Implemented:

- pure node
- no valid split
- maximum depth

Possible future additions:

- minimum samples per split
- minimum samples per leaf
- minimum impurity reduction
- post-pruning

---

### ✓ Prediction

**Status:** Complete

Implemented recursive prediction for individual observations.

The algorithm traverses the tree until reaching a leaf and returns the leaf's class prediction.

---

### ✓ Human-Readable Tree Output

**Status:** Complete

Implemented `print_tree()` with optional feature names.

Example:

```text
PetalLength <= 2.45
├── Yes:
    Predict: setosa
└── No:
    PetalLength <= 4.95
    ├── Yes:
        Predict: versicolor
    └── No:
        Predict: virginica
```

---

# Phase 3 — Evaluation

### ✓ Iris Dataset

**Status:** Complete

The implementation was tested using the classic Iris dataset:

```text
150 observations
4 numeric features
3 classes
```

Features:

- SepalLength
- SepalWidth
- PetalLength
- PetalWidth

Classes:

- setosa
- versicolor
- virginica

The unconstrained CART implementation successfully learned the Iris training data.

---

### ✓ Train/Test Split

**Status:** Complete

Implemented a reproducible 80/20 random split using:

```julia
Random.seed!(42)
```

Result:

```text
Training observations: 120
Test observations:      30
```

---

### ✓ Accuracy

**Status:** Complete

Accuracy is calculated as

$$
\text{Accuracy}
=
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
$$

For the unconstrained Iris tree:

```text
Training accuracy = 1.000
Test accuracy     = 0.967
```

The tree correctly classified 29 of the 30 test observations.

---

### ✓ Confusion Matrix

**Status:** Complete

Implemented a multiclass confusion matrix.

Rows represent actual classes and columns represent predicted classes.

The test set contained only one classification error: a `versicolor` observation was classified as `virginica`.

The misclassified observation was:

```text
SepalLength = 6.0
SepalWidth  = 2.7
PetalLength = 5.1
PetalWidth  = 1.6

Actual:    versicolor
Predicted: virginica
```

Tracing the observation through the tree demonstrated how CART's hard threshold boundaries can produce different classifications for observations with very similar feature values.

---

### ✓ Maximum-Depth Experiment

**Status:** Complete

The effect of `max_depth` was evaluated on the fixed Iris train/test split.

```text
Depth   Train Accuracy   Test Accuracy
1       0.683            0.600
2       0.950            0.933
3       0.983            0.967
4       1.000            0.967
5+      1.000            0.967
```

Depth 1 clearly underfits the data.

At depth 3, test accuracy reaches 0.967. Increasing the depth further improves training accuracy to 1.0 but does not improve test accuracy.

This provides a direct demonstration of the relationship between model complexity, training fit, and generalization.

---

### ✓ K-Fold Cross-Validation

**Status:** Complete

Implemented 5-fold cross-validation for selecting `max_depth`.

The 120-observation training set is divided into five folds. For each candidate depth:

1. Train on four folds.
2. Validate on the remaining fold.
3. Repeat until every fold has served as validation data.
4. Average the validation accuracies.

Results:

```text
Depth   Mean CV Accuracy
1       0.658
2       0.908
3       0.967
4       0.967
5       0.967
6       0.967
7       0.967
8       0.967
9       0.967
10      0.967
```

Depths 3 through 10 produced the same mean validation accuracy.

Because depth 3 achieves the same validation performance with a simpler tree, it is the natural model selected from these candidates.

The final workflow is therefore:

```text
Full dataset
     |
     +---- Training set
     |        |
     |     K-fold CV
     |        |
     |   select max_depth
     |        |
     |   retrain model
     |
     +---- Test set
              |
        final evaluation
```

The test set remains outside the model-selection process.

---

# Phase 4 — Extensions

### □ Additional Stopping Criteria

Potential additions:

- `min_samples_split`
- `min_samples_leaf`
- minimum impurity reduction

---

### □ Pruning

Explore cost-complexity pruning as an alternative to limiting tree depth during initial construction.

---

### □ Bagging

Train multiple trees using bootstrap samples of the training data.

Combine classification results using majority voting.

---

### □ Random Forest

Extend bagging by randomly selecting candidate feature subsets at each split.

A Random Forest will therefore introduce two major sources of randomness:

- bootstrap sampling of observations
- random feature subsampling

---

### □ Feature Subsampling

Randomly select a subset of features at each node rather than allowing every feature to compete for every split.

This reduces correlation among trees in a Random Forest.

---

# Development Log

## Milestone 1 — Impurity and Splitting

**Completed**

- Implemented Gini impurity.
- Implemented weighted Gini.
- Implemented candidate threshold generation.
- Implemented best split for one feature.
- Implemented search across all features.

### Key Insight

CART does not directly maximize classification accuracy while constructing the tree.

Instead, it greedily minimizes child-node impurity at each split.

---

## Milestone 2 — Recursive CART

**Completed**

- Implemented `TreeNode`.
- Implemented recursive tree construction.
- Implemented pure-node stopping.
- Implemented handling when no valid split exists.
- Implemented human-readable tree printing.

### Key Insight

The same tree-building operation is recursively applied to increasingly smaller subsets of the training data.

A trained decision tree is ultimately a hierarchy of learned `if/else` rules.

---

## Milestone 3 — Prediction and Evaluation

**Completed**

- Implemented prediction.
- Tested on the Iris dataset.
- Implemented reproducible train/test splitting.
- Implemented accuracy measurement.
- Implemented confusion matrix.
- Traced individual classification errors through the learned tree.

### Key Insight

Gini impurity is used during **training**, not prediction.

Prediction simply traverses the learned decision rules until reaching a leaf.

---

## Milestone 4 — Model Complexity and Validation

**Completed**

- Added `max_depth`.
- Compared training and test accuracy across tree depths.
- Implemented 5-fold cross-validation.
- Selected tree depth using validation performance.

### Key Insight

Increasing tree depth improves training fit but does not necessarily improve performance on unseen data.

Cross-validation provides a more reliable method for selecting model complexity than choosing parameters based on test-set performance.

---

# Current Status

The core CART classification implementation is complete.

The project currently supports:

- Gini impurity
- weighted Gini
- numeric threshold generation
- greedy split optimization
- recursive binary tree construction
- maximum-depth control
- prediction
- readable tree output
- train/test evaluation
- confusion matrices
- k-fold cross-validation

The next major development phase is to extend the single-tree implementation toward **bagging and Random Forests**, while also exploring additional stopping and pruning strategies.