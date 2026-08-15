function build_bagged_trees(X, y, n_trees::Int; max_depth=10, mtry=nothing)
    forest = []

    for b in 1:n_trees
        boot_idx, oob_idx = bootstrap_indices(length(y))

        X_boot = X[boot_idx, :]
        y_boot = y[boot_idx]

        tree = CART.build_tree(
            X_boot,
            y_boot;
            max_depth=max_depth,
            mtry=mtry
        )

        push!(forest, (tree=tree, oob_idx=oob_idx))
    end

    return forest
end

function majority_vote(predictions)
    classes = unique(predictions)

    best_class = classes[1]
    best_count = 0

    for c in classes
        count_c = count(==(c), predictions)

        if count_c > best_count
            best_count = count_c
            best_class = c
        end
    end

    return best_class
end

function predict_forest(trees, x)
    predictions = [
        Predict.predict(tree, x)
        for tree in trees
    ]

    return majority_vote(predictions)
end
function predict_oob(forest, X, i)
    predictions = Int[]

    for member in forest
        if i in member.oob_idx
            prediction = Predict.predict(
                member.tree,
                X[i, :]
            )

            push!(predictions, prediction)
        end
    end

    if isempty(predictions)
        return nothing
    end

    return majority_vote(predictions)
end