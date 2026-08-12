module Gini

export gini
export weighted_gini

"""
    gini(y)

Compute the Gini impurity of a vector of class labels.

Gini impurity:

    G = 1 - sum(p_k^2)

where p_k is the proportion of observations belonging to class k.
"""
function gini(y)
    n = length(y)

    if n == 0
        return 0.0
    end

    classes = unique(y)

    impurity = 1.0

    for c in classes
        p = count(==(c), y) / n
        impurity -= p^2
    end

    return impurity
end

"""
    weighted_gini(y_left, y_right)

Compute the weighted Gini impurity of a candidate split.
"""
function weighted_gini(y_left, y_right)

    n_left = length(y_left)
    n_right = length(y_right)
    n_total = n_left + n_right

    n_total == 0 && error("Both child nodes are empty.")

    g_left = gini(y_left)
    g_right = gini(y_right)

    return (n_left / n_total) * g_left +
           (n_right / n_total) * g_right
end

end