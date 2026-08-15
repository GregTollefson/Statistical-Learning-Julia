# Statistics supplies mean() and std(), which are used for accuracy summaries
# and for measuring run-to-run variation in the repeated experiment.
using Statistics

using Statistics

# Load the CART implementation developed in the parent directory. include()
# evaluates each source file in the current Julia session so that its modules
# and functions are available to this experiment script.
include("../Gini.jl")
include("../Node.jl")
include("../Split.jl")
include("../CART.jl")
include("../Predict.jl")

# Load the Random Forest-specific pieces: bootstrap sampling, the Wine dataset
# loader, and the ensemble/bagging functions.
include("Bootstrap.jl")
include("LoadWine.jl")
include("Bagging.jl")

# -----------------------------
# Load Wine dataset
# -----------------------------

# X is an n x p matrix of predictor values and y is the vector of class labels.
# For the UCI Wine data, n = 178 observations and p = 13 features.
X, y = load_wine("data/wine/wine.data")

# Print a quick sanity check before fitting any models. This confirms that the
# expected number of rows, features, and class labels were loaded.
println("Wine dataset loaded")
println("Observations: ", size(X, 1))
println("Features: ", size(X, 2))
println("Classes: ", unique(y))
println()

# -----------------------------
# Single CART
# -----------------------------

# Establish a baseline using one ordinary CART tree. No feature subsampling is
# requested here, and the tree is restricted to a maximum depth of four.
cart_tree = CART.build_tree(
    X,
    y;
    max_depth=4
)

# Generate one prediction for every row in X. This is TRAINING-set prediction:
# the same observations used to fit cart_tree are also being predicted here.
cart_pred = [
    Predict.predict(cart_tree, X[i, :])
    for i in 1:size(X, 1)
]

# Element-wise comparison produces a Boolean vector such as
# [true, true, false, ...]. In Julia, mean() treats true as 1 and false as 0,
# so the mean of this vector is the fraction classified correctly.
cart_accuracy = mean(cart_pred .== y)

println("Single CART")
println("Training accuracy: ", round(cart_accuracy, digits=4))
println()

# -----------------------------
# Bagged CART
# -----------------------------

# Build 100 CART trees, each from a different bootstrap sample.
#
# mtry=nothing means every predictor is available when CART searches for the
# best split. Thus this model uses bootstrap aggregation, but NOT Random Forest
# feature subsampling.
bagged_forest = build_bagged_trees(
    X,
    y,
    100;
    max_depth=4,
    mtry=nothing
)

# Obtain an OOB prediction for every original observation. For observation i,
# predict_oob() uses only trees whose bootstrap samples did not contain i.
bagged_oob_pred = [
    predict_oob(bagged_forest, X, i)
    for i in 1:size(X, 1)
]

# predict_oob() returns nothing if no tree has observation i out-of-bag.
# Create a mask that retains only observations for which an OOB prediction is
# available. With 100 trees, almost every observation should have many OOB
# voters, but explicitly checking makes the calculation robust.
bagged_valid = .!isnothing.(bagged_oob_pred)

# Select only valid OOB predictions, convert the resulting values to Int, and
# compare them with the corresponding true class labels. Unlike cart_accuracy,
# this is an internal validation estimate rather than training accuracy.
bagged_oob_accuracy = mean(
    Int.(bagged_oob_pred[bagged_valid]) .== y[bagged_valid]
)

println("Bagged CART")
println("OOB accuracy: ", round(bagged_oob_accuracy, digits=4))
println()

# -----------------------------
# Random Forest
# -----------------------------

# Build another 100-tree bootstrap ensemble, but now set mtry=3.
#
# At each CART split, only three randomly selected features are eligible for
# consideration. This extra source of randomness reduces correlation among
# trees, which is the defining extension from bagging to Random Forest.
rf_forest = build_bagged_trees(
    X,
    y,
    100;
    max_depth=4,
    mtry=3
)

# As with the bagged model, evaluate each observation using only trees for
# which that observation was out-of-bag.
rf_oob_pred = [
    predict_oob(rf_forest, X, i)
    for i in 1:size(X, 1)
]

# Retain observations that received at least one valid OOB prediction.
rf_valid = .!isnothing.(rf_oob_pred)

# Calculate Random Forest OOB classification accuracy.
rf_oob_accuracy = mean(
    Int.(rf_oob_pred[rf_valid]) .== y[rf_valid]
)

println("Random Forest")
println("OOB accuracy: ", round(rf_oob_accuracy, digits=4))
println()

# -----------------------------
# Summary
# -----------------------------

# Display the three results together. Note that the first number is training
# accuracy while the other two are OOB accuracies, so the CART number is not a
# like-for-like estimate of generalization performance.
println("Summary")
println("-----------------------------")
println(
    "CART training accuracy:       ",
    round(cart_accuracy, digits=4)
)
println(
    "Bagged CART OOB accuracy:     ",
    round(bagged_oob_accuracy, digits=4)
)
println(
    "Random Forest OOB accuracy:   ",
    round(rf_oob_accuracy, digits=4)
)

# -----------------------------
# Repeated OOB experiment
# -----------------------------

# Because bootstrap samples and Random Forest feature subsets are random, one
# run can make either method look slightly better simply due to sampling noise.
# This function repeats the complete Bagging-vs-Random-Forest comparison and
# summarizes both the average OOB accuracy and its run-to-run variability.
function run_repeated_experiment(X, y;
                                 n_runs=10,
                                 n_trees=100,
                                 max_depth=4,
                                 mtry=3)

    # Store one OOB accuracy from each independent run for each method.
    bagged_scores = Float64[]
    rf_scores = Float64[]

    for run in 1:n_runs

        # ----- Bagged CART -----
        # Every split may consider all features because mtry=nothing.
        bagged_forest = build_bagged_trees(
            X,
            y,
            n_trees;
            max_depth=max_depth,
            mtry=nothing
        )

        # Compute an OOB prediction for each observation using this run's
        # independently generated collection of bootstrap trees.
        bagged_oob_pred = [
            predict_oob(bagged_forest, X, i)
            for i in 1:size(X, 1)
        ]

        # Exclude any observation that happened to receive no OOB votes.
        bagged_valid = .!isnothing.(bagged_oob_pred)

        # Compute the OOB accuracy for this particular bagging run.
        bagged_acc = mean(
            Int.(bagged_oob_pred[bagged_valid]) .== y[bagged_valid]
        )

        # Save the score so that mean and standard deviation can be calculated
        # across all independent repetitions after the loop finishes.
        push!(bagged_scores, bagged_acc)

        # ----- Random Forest -----
        # The bootstrap mechanism is identical to bagging, but mtry restricts
        # the number of candidate features considered at each node split.
        rf_forest = build_bagged_trees(
            X,
            y,
            n_trees;
            max_depth=max_depth,
            mtry=mtry
        )

        # Generate OOB predictions using only eligible trees for each row.
        rf_oob_pred = [
            predict_oob(rf_forest, X, i)
            for i in 1:size(X, 1)
        ]

        # Again, protect against the possibility of observations with no OOB
        # voters before calculating accuracy.
        rf_valid = .!isnothing.(rf_oob_pred)

        # OOB accuracy for this particular Random Forest run.
        rf_acc = mean(
            Int.(rf_oob_pred[rf_valid]) .== y[rf_valid]
        )

        # Save the score for the across-run summary.
        push!(rf_scores, rf_acc)

        # Report progress and make it possible to see how much the two methods
        # fluctuate from one random realization to the next.
        println(
            "Run ",
            run,
            " | Bagging = ",
            round(bagged_acc, digits=4),
            " | RF = ",
            round(rf_acc, digits=4)
        )
    end

    # Summarize the empirical distribution of OOB accuracies over n_runs.
    # mean() estimates typical performance; std() measures how sensitive the
    # observed accuracy is to the random bootstrap/feature selections.
    println()
    println("Repeated Experiment Summary")
    println("-----------------------------")
    println(
        "Bagged CART mean OOB accuracy: ",
        round(mean(bagged_scores), digits=4)
    )
    println(
        "Bagged CART std:               ",
        round(std(bagged_scores), digits=4)
    )
    println(
        "Random Forest mean OOB accuracy: ",
        round(mean(rf_scores), digits=4)
    )
    println(
        "Random Forest std:               ",
        round(std(rf_scores), digits=4)
    )

    # Returning the raw scores preserves the individual runs for later plots,
    # hypothesis tests, or other diagnostics instead of returning only the
    # printed summary statistics.
    return bagged_scores, rf_scores
end