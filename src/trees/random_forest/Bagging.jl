function build_bagged_trees(X, y, n_trees::Int; max_depth=10)
    trees = []

    for b in 1:n_trees
        boot_idx, oob_idx = bootstrap_indices(length(y))

        X_boot = X[boot_idx, :]
        y_boot = y[boot_idx]

        tree = CART.build_tree(
            X_boot,
            y_boot;
            max_depth=max_depth
        )

        push!(trees, tree)
    end

    return trees
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