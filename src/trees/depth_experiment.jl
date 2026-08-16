# depth_experiment.jl
#
# Examine how maximum tree depth affects training and test accuracy.
#
# This script assumes setup.jl has already created X_train, y_train, X_test,
# y_test and made build_tree() / predict() available in the current Julia
# session.
#
# Statistical idea:
#   - shallow trees may underfit because they cannot represent enough structure
#   - very deep trees can fit the training data extremely well and may overfit
#   - comparing train and test accuracy across depths helps visualize this
#     bias/variance tradeoff

# Evaluate integer depth limits from 1 through 10.
depths = 1:10

# Store one training and one test accuracy value for each depth so the results
# remain available after the loop for plotting or further analysis.
train_accuracies = Float64[]
test_accuracies = Float64[]

for depth in depths

    # Train a fresh CART model using the same training data but a different
    # maximum allowed tree depth.
    tree_depth = build_tree(
        X_train,
        y_train;
        max_depth=depth
    )

    # Generate predictions for every observation used to fit the tree.
    # This measures how closely the model can reproduce the training sample.
    y_pred_train_depth = [
        predict(tree_depth, X_train[i, :])
        for i in 1:size(X_train, 1)
    ]

    # Generate predictions on held-out observations that the tree did not see
    # during fitting. Test accuracy is therefore the more relevant estimate of
    # generalization performance in this experiment.
    y_pred_test_depth = [
        predict(tree_depth, X_test[i, :])
        for i in 1:size(X_test, 1)
    ]

    # Element-wise equality produces a Boolean vector. sum() counts the `true`
    # values because Julia treats true as 1 and false as 0 in this arithmetic
    # context; dividing by the number of observations gives classification
    # accuracy.
    train_acc =
        sum(y_pred_train_depth .== y_train) / length(y_train)

    test_acc =
        sum(y_pred_test_depth .== y_test) / length(y_test)

    # Preserve the results in vectors whose positions correspond to `depths`.
    push!(train_accuracies, train_acc)
    push!(test_accuracies, test_acc)

    # Print each result immediately so the trend can be inspected directly in
    # the REPL without requiring a plot.
    println(
        "Depth = ", depth,
        " | Train = ", round(train_acc, digits=3),
        " | Test = ", round(test_acc, digits=3)
    )
end