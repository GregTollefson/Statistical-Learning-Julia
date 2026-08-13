# CART Decision Tree Implementation Journal (Julia)

## Project Overview

**Project:** CART (Classification and Regression Trees) from First Principles

**Language:** Julia

**Objective:** Build a complete CART implementation without relying on machine learning libraries in order to understand the underlying algorithms and mathematics.

This project is intended as both a software engineering exercise and a statistical learning exercise. Each component will be implemented incrementally, tested independently, and documented before integrating into the complete tree-building algorithm.

---
# Mathematical Background

The Classification and Regression Tree (CART) algorithm builds a decision tree by recursively partitioning the feature space into increasingly homogeneous regions. At each node, the algorithm searches for the feature and threshold that produces the "best" split according to an impurity measure.

Unlike many statistical methods that optimize a global objective directly, CART follows a **greedy optimization strategy**. Each split is chosen solely to maximize the immediate reduction in impurity at the current node. Although this does not guarantee the globally optimal tree, it produces highly effective models while remaining computationally tractable.

---

## Node Impurity

For classification problems, CART measures the impurity of a node using the **Gini Impurity**.

Suppose a node contains observations from $K$ classes. Let

$$
p_k = \frac{\text{Number of observations in class }k}
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
- The maximum impurity occurs when all classes are equally represented.

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

## Recursive Partitioning

After selecting the best split, the same procedure is applied independently to each child node.

This recursive process continues until one or more stopping criteria are met, such as

- the node is pure,
- the maximum tree depth is reached,
- too few observations remain,
- or no split produces a meaningful reduction in impurity.

The result is a binary decision tree that approximates the underlying classification function.

---

# Roadmap

## Phase 1 — Foundations

### ✓ Gini Impurity

**Status:** Complete

Implemented the Gini impurity function

$$
G = 1 - \sum_{k=1}^{K} p_k^2
$$

where $p_k$ is the proportion of observations belonging to class $k$.

### Learning Notes

- Gini measures node impurity.
- A pure node has $G = 0$.
- Maximum impurity occurs when classes are evenly distributed.
- Gini is used as the optimization criterion when selecting splits.

---

### □ Weighted Gini

**Status:** Next Task

Implement the impurity of a candidate split

$$
G_{\text{split}}
=
\frac{n_L}{n}G_L
+
\frac{n_R}{n}G_R
$$

where

- $G_L$ = left child impurity
- $G_R$ = right child impurity
- $n_L, n_R$ = observations in each child

The best split minimizes this quantity.

---

### □ Candidate Threshold Generation

Generate all possible split locations for a feature.

Example

Feature values

```text
1
2
5
7
9
```

Candidate thresholds

```text
1.5
3.5
6
8
```

These midpoints become the possible decision boundaries.

---

### □ Best Split for One Feature

For every candidate threshold:

1. Divide observations into left/right nodes.
2. Compute weighted Gini.
3. Keep the threshold with the lowest impurity.

Output:

- Best threshold
- Gini score
- Left indices
- Right indices

---

# Phase 2 — CART Algorithm

### □ Search Across All Features

Repeat the single-feature split search for every feature.

Choose the feature/threshold pair with the minimum weighted Gini.

---

### □ Node Structure

Define a node that stores

- prediction
- feature index
- threshold
- left child
- right child
- depth
- impurity
- sample count

---

### □ Recursive Tree Construction

Algorithm

1. Find best split.
2. Create child nodes.
3. Recurse on each child.
4. Stop when stopping criteria are met.

---

### □ Stopping Criteria

Possible stopping rules

- maximum depth
- minimum samples
- pure node
- no improvement in impurity

---

### □ Prediction

Traverse the tree

```julia
if x[j] <= threshold
    go left
else
    go right
```

Return the class prediction at the leaf.

---

# Phase 3 — Evaluation

### □ Train/Test Split

Randomly divide the dataset into training and testing subsets.

---

### □ Confusion Matrix

Evaluate

- True Positive
- False Positive
- True Negative
- False Negative

---

### □ Accuracy

Compute

$$
\text{Accuracy}
=
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
$$

---

### □ Tree Visualization

Display the learned tree in a readable format.

Example

```text
PetalLength <= 2.45

├── True
│     Predict: Setosa
│
└── False
      PetalWidth <= 1.75

      ├── True
      │      Predict: Versicolor
      │
      └── False
             Predict: Virginica
```

---

# Phase 4 — Extensions

### □ Bagging

Bootstrap sampling of training data.

---

### □ Random Forest

Construct multiple decision trees using

- bootstrap datasets
- random feature subsets

Combine predictions by majority vote.

---

### □ Feature Subsampling

Randomly select a subset of features at every split to reduce correlation between trees.

---

# Development Log

## Milestone 1

**Completed**

- Project structure defined.
- Implemented Gini impurity function.
- Verified against hand calculations.

### Key Insight

A decision tree does **not** directly maximize accuracy.

Instead, it greedily minimizes node impurity at every split, using Gini as a proxy for producing purer child nodes.

---

## Upcoming Milestone

Implement the weighted Gini function and use it to evaluate candidate splits for a single feature.

This will complete the mathematical foundation required before building the recursive CART algorithm.
