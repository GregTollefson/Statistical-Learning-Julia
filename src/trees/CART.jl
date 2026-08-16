# CART.jl implements recursive construction of the classification tree.
# Split.jl decides *where* to split; Node.jl provides the data structure used
# to store the resulting tree.
module CART

# Import only the specific functions/types required by this module.
using ..Split: best_split
using ..Node: TreeNode

export build_tree, print_tree

# Return the most common class label in y.
#
# CART uses this in two situations:
#   1. when a stopping rule turns the current node into a leaf, and
#   2. to store a representative prediction in every internal TreeNode.
function majority_class(y)
    # unique(y) gives the distinct class labels appearing at this node.
    classes = unique(y)

    # Initialize the running winner with the first observed class.
    best_class = classes[1]
    best_count = 0

    for c in classes
        # ==(c) is a predicate equivalent to x -> x == c.
        # count() therefore counts the observations belonging to class c.
        n = count(==(c), y)

        # Retain the class with the largest count encountered so far.
        if n > best_count
            best_count = n
            best_class = c
        end
    end

    return best_class
end

# Recursively build a classification tree from feature matrix X and labels y.
#
# Keyword arguments:
#   max_depth -> optional limit on tree depth; `nothing` means no explicit limit
#   depth     -> current recursive depth, starting at zero for the root node
#   mtry      -> number of randomly selected candidate features per split;
#                `nothing` gives standard CART, while an integer enables the
#                Random Forest-style feature-subsampling behavior
function build_tree(X, y; max_depth=nothing, depth=0, mtry=nothing)

    # ---------------------------------------------------------------
    # Stopping rule 1: pure node
    # ---------------------------------------------------------------
    # If only one unique class remains, no further split can improve purity.
    # The node becomes a leaf predicting that sole class.
    if length(unique(y)) == 1
        return TreeNode(y[1], nothing, nothing, nothing, nothing)
    end

    # ---------------------------------------------------------------
    # Stopping rule 2: maximum depth reached
    # ---------------------------------------------------------------
    # depth >= max_depth prevents additional recursive splitting. Because the
    # node may still contain several classes, predict the local majority class.
    if max_depth !== nothing && depth >= max_depth
        return TreeNode(
            majority_class(y),
            nothing,
            nothing,
            nothing,
            nothing
        )
    end

    # Ask Split.best_split() to search the eligible features and candidate
    # thresholds for the partition having minimum weighted Gini impurity.
    #
    # `score` is returned for diagnostic completeness even though build_tree()
    # does not need it after the winning split has been identified.
    feature, threshold, score = best_split(X, y; mtry=mtry)

    # ---------------------------------------------------------------
    # Stopping rule 3: no valid split exists
    # ---------------------------------------------------------------
    # This can occur when the remaining predictor values cannot partition the
    # observations any further. Convert the node to a majority-class leaf.
    if feature === nothing
        return TreeNode(
            majority_class(y),
            nothing,
            nothing,
            nothing,
            nothing
        )
    end

    # Apply the selected decision rule to every row in X.
    # Observations satisfying X[:, feature] <= threshold go left; the rest go
    # right. Broadcasting (`.<=` and `.>`) produces Boolean masks row by row.
    left_mask = X[:, feature] .<= threshold
    right_mask = X[:, feature] .> threshold

    # Construct the data subset passed recursively to the left child.
    X_left = X[left_mask, :]
    y_left = y[left_mask]

    # Construct the data subset passed recursively to the right child.
    X_right = X[right_mask, :]
    y_right = y[right_mask]

    # Recursively build the entire left subtree. Notice that depth increases by
    # one while max_depth and mtry are propagated unchanged.
    left_child = build_tree(
        X_left,
        y_left;
        max_depth=max_depth,
        depth=depth + 1,
        mtry=mtry
    )

    # Recursively build the entire right subtree using the complementary data.
    right_child = build_tree(
        X_right,
        y_right;
        max_depth=max_depth,
        depth=depth + 1,
        mtry=mtry
    )

    # Once both subtrees are complete, assemble the current internal node.
    # The node stores the split rule plus references to its two child trees.
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

    # Repeat four spaces once for every level below the root so that recursive
    # tree structure appears visually indented in terminal output.
    indent = "    " ^ depth

    # A node with no feature index is a leaf. Print its predicted class and end
    # this recursive branch immediately.
    if node.feature === nothing
        println(indent, "Predict: ", node.prediction)
        return
    end

    # Convert the stored integer feature index into something readable. If no
    # feature_names vector was supplied, fall back to a generic label.
    if feature_names === nothing
        feature_label = "Feature $(node.feature)"
    else
        feature_label = feature_names[node.feature]
    end

    # Print the decision rule represented by this internal node.
    println(
        indent,
        feature_label,
        " <= ",
        node.threshold
    )

    # Recursively print the branch followed when the split condition is true.
    println(indent, "├── Yes:")
    print_tree(node.left, feature_names, depth + 1)

    # Recursively print the branch followed when the split condition is false.
    println(indent, "└── No:")
    print_tree(node.right, feature_names, depth + 1)
end

end