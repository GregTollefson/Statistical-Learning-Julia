# Define a module so the impurity functions live in their own namespace.
# Other files can then refer to them explicitly as Gini.gini() or import
# selected functions with `using ..Gini` / `using ..Gini: weighted_gini`.
module Gini

# Make these two functions available to code that imports this module.
export gini
export weighted_gini

"""
    gini(y)

Compute the Gini impurity of a vector of class labels.

Gini impurity:

    G = 1 - sum(p_k^2)

where p_k is the proportion of observations belonging to class k.

Interpretation:
- G = 0 means the node is pure: every observation belongs to one class.
- Larger values mean the class labels are more mixed.
- CART searches for splits that produce child nodes with low impurity.
"""
function gini(y)
    # Number of observations currently contained in this node.
    n = length(y)

    # An empty child node is assigned impurity zero. This also prevents a
    # division-by-zero error when class proportions are calculated below.
    if n == 0
        return 0.0
    end

    # unique(y) returns one copy of each class label present in this node.
    # For example, ["A", "A", "B"] becomes ["A", "B"].
    classes = unique(y)

    # Start with 1 and subtract p_k^2 for each class, implementing
    #     G = 1 - Σ p_k^2.
    impurity = 1.0

    for c in classes
        # ==(c) creates a function equivalent to x -> x == c.
        # count(==(c), y) therefore counts how many labels in y equal c.
        # Dividing by n converts that count into the empirical class
        # probability p_k at this node.
        p = count(==(c), y) / n

        # Subtract this class's squared probability from the running total.
        # Squaring makes large class proportions contribute more strongly;
        # a node dominated by one class therefore has lower impurity.
        impurity -= p^2
    end

    return impurity
end

"""
    weighted_gini(y_left, y_right)

Compute the weighted Gini impurity of a candidate split.

The two child impurities are weighted by the fraction of observations sent to
that child. A split is considered better when this weighted value is smaller.
"""
function weighted_gini(y_left, y_right)

    # Count how many observations the proposed split sends to each child.
    n_left = length(y_left)
    n_right = length(y_right)
    n_total = n_left + n_right

    # There is no meaningful impurity calculation if both children are empty.
    # The short-circuit form `condition && expression` evaluates the error only
    # when n_total == 0.
    n_total == 0 && error("Both child nodes are empty.")

    # Compute impurity independently for the left and right child nodes.
    g_left = gini(y_left)
    g_right = gini(y_right)

    # Weight each child by its relative size. A tiny pure child should not
    # dominate the score if the other child contains almost all observations.
    #
    #     G_split = (n_L / n) G_L + (n_R / n) G_R
    return (n_left / n_total) * g_left +
           (n_right / n_total) * g_right
end

end