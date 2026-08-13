# setup.jl

# -------------------------------------------------
# Load CART modules
# -------------------------------------------------

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

using RDatasets
using Random


# -------------------------------------------------
# Load Iris dataset
# -------------------------------------------------

iris = dataset("datasets", "iris")

X = Matrix(iris[:, 1:4])
y = String.(iris[:, 5])

feature_names = names(iris)[1:4]


# -------------------------------------------------
# Reproducible 80/20 train-test split
# -------------------------------------------------

Random.seed!(42)

indices = shuffle(1:size(X, 1))

n_train = round(Int, 0.8 * length(indices))

train_idx = indices[1:n_train]
test_idx = indices[n_train+1:end]

X_train = X[train_idx, :]
y_train = y[train_idx]

X_test = X[test_idx, :]
y_test = y[test_idx]


# -------------------------------------------------
# Build CART model
# -------------------------------------------------

tree = build_tree(X_train, y_train)


# -------------------------------------------------
# Generate predictions
# -------------------------------------------------

y_pred_train = [
    predict(tree, X_train[i, :])
    for i in 1:size(X_train, 1)
]

y_pred_test = [
    predict(tree, X_test[i, :])
    for i in 1:size(X_test, 1)
]


# -------------------------------------------------
# Accuracy
# -------------------------------------------------

train_accuracy =
    sum(y_pred_train .== y_train) / length(y_train)

test_accuracy =
    sum(y_pred_test .== y_test) / length(y_test)


# -------------------------------------------------
# Status
# -------------------------------------------------

println("CART development environment loaded.")
println("Iris dataset: ", size(X, 1), " observations, ",
        size(X, 2), " features")
println("Training observations: ", length(y_train))
println("Test observations: ", length(y_test))
println("Training accuracy: ", train_accuracy)
println("Test accuracy: ", test_accuracy)