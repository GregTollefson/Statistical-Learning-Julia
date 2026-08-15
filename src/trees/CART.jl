module CART

using ..Split: best_split
using ..Node: TreeNode

export build_tree, print_tree

function majority_class(y)
    classes = unique(y)

    best_class = classes[1]
    best_count = 0

    for c in classes
        n = count(==(c), y)

        if n > best_count
            best_count = n
            best_class = c
        end
    end

    return best_class
end


function build_tree(X, y; max_depth=nothing, depth=0, mtry=nothing)

    # Pure node
    if length(unique(y)) == 1
        return TreeNode(y[1], nothing, nothing, nothing, nothing)
    end

    # Maximum depth reached
    if max_depth !== nothing && depth >= max_depth
        return TreeNode(
            majority_class(y),
            nothing,
            nothing,
            nothing,
            nothing
        )
    end

    feature, threshold, score = best_split(X, y; mtry=mtry)

    if feature === nothing
        return TreeNode(
            majority_class(y),
            nothing,
            nothing,
            nothing,
            nothing
        )
    end

    left_mask = X[:, feature] .<= threshold
    right_mask = X[:, feature] .> threshold

    X_left = X[left_mask, :]
    y_left = y[left_mask]

    X_right = X[right_mask, :]
    y_right = y[right_mask]

    left_child = build_tree(
        X_left,
        y_left;
        max_depth=max_depth,
        depth=depth + 1,
        mtry=mtry
    )

    right_child = build_tree(
        X_right,
        y_right;
        max_depth=max_depth,
        depth=depth + 1,
        mtry=mtry
    )

    return TreeNode(
        majority_class(y),
        feature,
        threshold,
        left_child,
        right_child
    )
end

"""
    print_tree(node, feature_names=nothing, depth=0)

Print a decision tree in a human-readable form.

Optionally provide feature names for more descriptive output.
"""
function print_tree(node::TreeNode, feature_names=nothing, depth=0)

    indent = "    " ^ depth

    # Leaf node
    if node.feature === nothing
        println(indent, "Predict: ", node.prediction)
        return
    end

    # Determine how to display the feature
    if feature_names === nothing
        feature_label = "Feature $(node.feature)"
    else
        feature_label = feature_names[node.feature]
    end

    # Decision node
    println(
        indent,
        feature_label,
        " <= ",
        node.threshold
    )

    # Left branch
    println(indent, "├── Yes:")
    print_tree(node.left, feature_names, depth + 1)

    # Right branch
    println(indent, "└── No:")
    print_tree(node.right, feature_names, depth + 1)
end

end