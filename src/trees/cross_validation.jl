# Random provides seed!() and shuffle(), used here to create reproducible folds.
using Random

# Estimate how well different maximum tree depths generalize using k-fold
# cross-validation.
#
# For each candidate depth:
#   1. divide the data into k folds
#   2. hold out one fold for validation
#   3. train on the remaining k-1 folds
#   4. repeat until every fold has served as validation data
#   5. average the k validation accuracies
#
# Unlike the separate train/test experiment, cross-validation reuses the
# available observations efficiently while still ensuring that each validation
# prediction is made by a tree that was not trained on that observation.
function cross_validate_depth(X, y; depths=1:10, k=5, seed=42)

    # Fix Julia's random-number generator so repeated calls with the same seed
    # produce the same shuffled ordering and therefore the same folds.
    Random.seed!(seed)

    # Number of observations is the number of rows in X.
    n = size(X, 1)

    # Randomly reorder observation indices before assigning them to folds.
    indices = shuffle(1:n)

    # Assign observations to approximately equal folds by taking every kth
    # shuffled index. For k=5, fold 1 receives positions 1,6,11,...; fold 2
    # receives 2,7,12,... and so on.
    folds = [indices[i:k:end] for i in 1:k]

    # Map each tested depth to its mean cross-validation accuracy.
    results = Dict{Int, Float64}()

    for depth in depths

        # Collect one validation accuracy from each of the k folds.
        fold_accuracies = Float64[]

        for fold in 1:k

            # The current fold acts as held-out validation data.
            val_idx = folds[fold]

            # Concatenate all other folds to obtain training indices. The
            # comprehension excludes the current validation fold, and `...`
            # splats the resulting vectors into vcat().
            train_idx = vcat(
                [folds[j] for j in 1:k if j != fold]...
            )

            # Construct this fold's training subset.
            X_cv_train = X[train_idx, :]
            y_cv_train = y[train_idx]

            # Construct this fold's validation subset.
            X_cv_val = X[val_idx, :]
            y_cv_val = y[val_idx]

            # Fit a new tree using only the k-1 training folds and the current
            # candidate maximum depth.
            tree_cv = build_tree(
                X_cv_train,
                y_cv_train;
                max_depth=depth
            )

            # Predict every observation in the held-out fold.
            y_pred = [
                predict(tree_cv, X_cv_val[i, :])
                for i in 1:size(X_cv_val, 1)
            ]

            # Count correct validation predictions and divide by fold size.
            accuracy =
                sum(y_pred .== y_cv_val) / length(y_cv_val)

            # Save this fold's score for averaging after all k folds are run.
            push!(fold_accuracies, accuracy)
        end

        # Average the k held-out accuracies to estimate performance at this
        # maximum depth.
        mean_accuracy = sum(fold_accuracies) / k
        results[depth] = mean_accuracy

        println(
            "Depth = ", depth,
            " | CV accuracy = ",
            round(mean_accuracy, digits=3)
        )
    end

    # Return the entire depth -> accuracy mapping so the caller can select the
    # best depth or plot the cross-validation curve.
    return results
end