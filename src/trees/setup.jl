# setup.jl
#
# Development setup script for the CART implementation.
#
# Running this file loads the tree modules, loads the Iris dataset, creates a
# reproducible train/test split, fits one CART model, generates predictions,
# and prints a compact status summary. The resulting variables remain available
# in the Julia session for experiments such as depth_experiment.jl and
# cross_validation.jl.

# -------------------------------------------------
# Load CART modules
# -------------------------------------------------

# include() evaluates the source file in the current Julia session. The
# following `using .ModuleName` statement then imports the exported names from
# that locally defined module; the leading dot refers to the current namespace.
include("Gini.jl")
using .Gini

include("Split.jl")
using .Split

include("Node.jl")
using .Node

include("CART.jl")
using .CART

include("Predict.jl")
using .Predict

include("Metrics.jl")
using .Metrics


# -------------------------------------------------
# Load packages
# -------------------------------------------------

# RDatasets provides convenient access to classic statistical datasets,
# including Fisher's Iris dataset.
using RDatasets

# Random supplies seed!() and shuffle() for reproducible data splitting.
using Random


# -------------------------------------------------
# Load Iris dataset
# -------------------------------------------------

# Retrieve the Iris dataset from R's standard `datasets` collection.
iris = dataset("datasets", "iris")

# The first four columns are numeric flower measurements. Matrix() converts the
# selected DataFrame columns into the dense feature matrix expected by CART.
X = Matrix(iris[:, 1:4])

# The fifth column contains species labels. Explicitly convert them to String so
# the classification code is exercised with non-integer class labels.
y = String.(iris[:, 5])

# Save human-readable predictor names for tree-printing and interpretation.
feature_names = names(iris)[1:4]


# -------------------------------------------------
# Reproducible 80/20 train-test split
# -------------------------------------------------

# Fix the random seed so the same observations enter the train and test sets
# each time setup.jl is run. This makes experiments directly comparable.
Random.seed!(42)

# Shuffle observation indices instead of rearranging X and y independently,
# ensuring feature rows stay aligned with their corresponding class labels.
indices = shuffle(1:size(X, 1))

# Use approximately 80% of the shuffled observations for model fitting.
n_train = round(Int, 0.8 * length(indices))

# The first block of shuffled indices becomes training data and the remainder
# becomes held-out test data.
train_idx = indices[1:n_train]
test_idx = indices[n_train+1:end]

# Construct the actual training feature matrix and class-label vector.
X_train = X[train_idx, :]
y_train = y[train_idx]

# Construct the held-out test set using exactly the same row indices for X/y.
X_test = X[test_idx, :]
y_test = y[test_idx]


# -------------------------------------------------
# Build CART model
# -------------------------------------------------

# Fit a full CART tree to the training set. Because max_depth is not supplied,
# build_tree() continues splitting until one of its intrinsic stopping rules is
# reached (for example, a pure node or no valid split).
tree = build_tree(X_train, y_train)


# -------------------------------------------------
# Generate predictions
# -------------------------------------------------

# Predict every training observation. These predictions measure fit to data the
# tree has already seen and therefore should not be treated as an unbiased
# estimate of generalization performance.
y_pred_train = [
    predict(tree, X_train[i, :])
    for i in 1:size(X_train, 1)
]

# Predict the held-out test observations. These rows were excluded from tree
# construction, so test accuracy provides a more meaningful performance check.
y_pred_test = [
    predict(tree, X_test[i, :])
    for i in 1:size(X_test, 1)
]


# -------------------------------------------------
# Accuracy
# -------------------------------------------------

# `.==` compares predictions and labels element by element. sum() counts the
# true values, and division by sample size converts that count to a proportion.
train_accuracy =
    sum(y_pred_train .== y_train) / length(y_train)

test_accuracy =
    sum(y_pred_test .== y_test) / length(y_test)


# -------------------------------------------------
# Status
# -------------------------------------------------

# Print enough information to verify that the environment, dataset split, model
# fit, and prediction path all executed successfully.
println("CART development environment loaded.")
println("Iris dataset: ", size(X, 1), " observations, ",
        size(X, 2), " features")
println("Training observations: ", length(y_train))
println("Test observations: ", length(y_test))
println("Training accuracy: ", train_accuracy)
println("Test accuracy: ", test_accuracy)