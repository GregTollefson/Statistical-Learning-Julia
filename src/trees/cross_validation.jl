using Random

function cross_validate_depth(X, y; depths=1:10, k=5, seed=42)

    Random.seed!(seed)

    n = size(X, 1)
    indices = shuffle(1:n)

    # Assign observations to approximately equal folds
    folds = [indices[i:k:end] for i in 1:k]

    results = Dict{Int, Float64}()

    for depth in depths

        fold_accuracies = Float64[]

        for fold in 1:k

            val_idx = folds[fold]

            train_idx = vcat(
                [folds[j] for j in 1:k if j != fold]...
            )

            X_cv_train = X[train_idx, :]
            y_cv_train = y[train_idx]

            X_cv_val = X[val_idx, :]
            y_cv_val = y[val_idx]

            tree_cv = build_tree(
                X_cv_train,
                y_cv_train;
                max_depth=depth
            )

            y_pred = [
                predict(tree_cv, X_cv_val[i, :])
                for i in 1:size(X_cv_val, 1)
            ]

            accuracy =
                sum(y_pred .== y_cv_val) / length(y_cv_val)

            push!(fold_accuracies, accuracy)
        end

        mean_accuracy = sum(fold_accuracies) / k
        results[depth] = mean_accuracy

        println(
            "Depth = ", depth,
            " | CV accuracy = ",
            round(mean_accuracy, digits=3)
        )
    end

    return results
end