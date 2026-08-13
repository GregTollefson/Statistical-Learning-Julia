# depth_experiment.jl
#
# Examine how maximum tree depth affects
# training and test accuracy.

depths = 1:10

train_accuracies = Float64[]
test_accuracies = Float64[]

for depth in depths

    # Train tree with specified maximum depth
    tree_depth = build_tree(
        X_train,
        y_train;
        max_depth=depth
    )

    # Training predictions
    y_pred_train_depth = [
        predict(tree_depth, X_train[i, :])
        for i in 1:size(X_train, 1)
    ]

    # Test predictions
    y_pred_test_depth = [
        predict(tree_depth, X_test[i, :])
        for i in 1:size(X_test, 1)
    ]

    # Calculate accuracy
    train_acc =
        sum(y_pred_train_depth .== y_train) / length(y_train)

    test_acc =
        sum(y_pred_test_depth .== y_test) / length(y_test)

    # Save results
    push!(train_accuracies, train_acc)
    push!(test_accuracies, test_acc)

    println(
        "Depth = ", depth,
        " | Train = ", round(train_acc, digits=3),
        " | Test = ", round(test_acc, digits=3)
    )
end