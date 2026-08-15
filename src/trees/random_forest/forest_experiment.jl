using Statistics

using Statistics

include("../Gini.jl")
include("../Node.jl")
include("../Split.jl")
include("../CART.jl")
include("../Predict.jl")

include("Bootstrap.jl")
include("LoadWine.jl")
include("Bagging.jl")

# -----------------------------
# Load Wine dataset
# -----------------------------

X, y = load_wine("data/wine/wine.data")

println("Wine dataset loaded")
println("Observations: ", size(X, 1))
println("Features: ", size(X, 2))
println("Classes: ", unique(y))
println()

# -----------------------------
# Single CART
# -----------------------------

cart_tree = CART.build_tree(
    X,
    y;
    max_depth=4
)

cart_pred = [
    Predict.predict(cart_tree, X[i, :])
    for i in 1:size(X, 1)
]

cart_accuracy = mean(cart_pred .== y)

println("Single CART")
println("Training accuracy: ", round(cart_accuracy, digits=4))
println()

# -----------------------------
# Bagged CART
# -----------------------------

bagged_forest = build_bagged_trees(
    X,
    y,
    100;
    max_depth=4,
    mtry=nothing
)

bagged_oob_pred = [
    predict_oob(bagged_forest, X, i)
    for i in 1:size(X, 1)
]

bagged_valid = .!isnothing.(bagged_oob_pred)

bagged_oob_accuracy = mean(
    Int.(bagged_oob_pred[bagged_valid]) .== y[bagged_valid]
)

println("Bagged CART")
println("OOB accuracy: ", round(bagged_oob_accuracy, digits=4))
println()

# -----------------------------
# Random Forest
# -----------------------------

rf_forest = build_bagged_trees(
    X,
    y,
    100;
    max_depth=4,
    mtry=3
)

rf_oob_pred = [
    predict_oob(rf_forest, X, i)
    for i in 1:size(X, 1)
]

rf_valid = .!isnothing.(rf_oob_pred)

rf_oob_accuracy = mean(
    Int.(rf_oob_pred[rf_valid]) .== y[rf_valid]
)

println("Random Forest")
println("OOB accuracy: ", round(rf_oob_accuracy, digits=4))
println()

# -----------------------------
# Summary
# -----------------------------

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

function run_repeated_experiment(X, y;
                                 n_runs=10,
                                 n_trees=100,
                                 max_depth=4,
                                 mtry=3)

    bagged_scores = Float64[]
    rf_scores = Float64[]

    for run in 1:n_runs

        bagged_forest = build_bagged_trees(
            X,
            y,
            n_trees;
            max_depth=max_depth,
            mtry=nothing
        )

        bagged_oob_pred = [
            predict_oob(bagged_forest, X, i)
            for i in 1:size(X, 1)
        ]

        bagged_valid = .!isnothing.(bagged_oob_pred)

        bagged_acc = mean(
            Int.(bagged_oob_pred[bagged_valid]) .== y[bagged_valid]
        )

        push!(bagged_scores, bagged_acc)

        rf_forest = build_bagged_trees(
            X,
            y,
            n_trees;
            max_depth=max_depth,
            mtry=mtry
        )

        rf_oob_pred = [
            predict_oob(rf_forest, X, i)
            for i in 1:size(X, 1)
        ]

        rf_valid = .!isnothing.(rf_oob_pred)

        rf_acc = mean(
            Int.(rf_oob_pred[rf_valid]) .== y[rf_valid]
        )

        push!(rf_scores, rf_acc)

        println(
            "Run ",
            run,
            " | Bagging = ",
            round(bagged_acc, digits=4),
            " | RF = ",
            round(rf_acc, digits=4)
        )
    end

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

    return bagged_scores, rf_scores
end