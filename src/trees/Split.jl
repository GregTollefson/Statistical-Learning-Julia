# The Split module contains the search logic that turns CART into a greedy
# optimization algorithm: at each node, evaluate candidate feature/threshold
# pairs and choose the split with the lowest weighted Gini impurity.
module Split

# randperm() is used by Random Forest to choose a random subset of features.
using Random: randperm

# Import only the impurity function needed by this module from the parent-level
# Gini module. The leading `..` refers to the enclosing module namespace.
using ..Gini: weighted_gini

export candidate_thresholds
export best_split_feature
export best_split

"""
    candidate_thresholds(x)

Generate candidate split thresholds for a feature vector.

The thresholds are the midpoints between consecutive unique,
sorted feature values.
"""
function candidate_thresholds(x)

    # CART only needs to consider thresholds between distinct observed values.
    # unique() removes duplicates and sort() puts the remaining values in
    # ascending order.
    values = sort(unique(x))

    # If every observation has the same value for this feature, no split is
    # possible because every threshold would send all observations to one side.
    if length(values) < 2
        return Float64[]
    end

    # Store candidate thresholds explicitly as floating-point values.
    thresholds = Float64[]

    # For consecutive sorted values a and b, the midpoint (a+b)/2 produces the
    # same partition as any threshold strictly between a and b. Therefore one
    # midpoint is sufficient to represent every distinct possible partition.
    for i in 1:(length(values) - 1)
        threshold = (values[i] + values[i + 1]) / 2
        push!(thresholds, threshold)
    end

    return thresholds
end

"""
    best_split_feature(x, y)

Find the best split threshold for a single feature vector `x`
using the class labels `y`.

Returns the threshold with the lowest weighted Gini impurity,
along with the corresponding impurity score.
"""
function best_split_feature(x, y)

    # Each feature value must correspond to exactly one class label.
    # `||` short-circuits: error() is called only when the lengths differ.
    length(x) == length(y) ||
        error("x and y must contain the same number of observations.")

    # Generate all distinct partitions that need to be evaluated for this
    # feature.
    thresholds = candidate_thresholds(x)

    # Start with no selected threshold and an infinite score. Since every real
    # weighted Gini score is finite and nonnegative, the first valid candidate
    # will automatically become the current best split.
    best_threshold = nothing
    best_score = Inf

    for threshold in thresholds

        # Broadcasting with `.<=` performs the comparison element by element.
        # The result is a Boolean vector indicating which observations belong
        # in the left child under this candidate split.
        left_mask = x .<= threshold

        # Observations strictly greater than the threshold form the right child.
        right_mask = x .> threshold

        # Boolean indexing selects only the class labels assigned to each child.
        y_left = y[left_mask]
        y_right = y[right_mask]

        # Lower weighted impurity means the candidate split produces purer
        # children and is therefore preferred by CART.
        score = weighted_gini(y_left, y_right)

        # Greedily retain the best threshold encountered so far.
        if score < best_score
            best_score = score
            best_threshold = threshold
        end
    end

    return best_threshold, best_score
end

"""
    best_split(X, y; mtry=nothing)

Find the best split across candidate features in the feature matrix `X`.

If `mtry` is `nothing`, all features are considered (standard CART).

If `mtry` is an integer, a random subset of `mtry` features is selected
and only those features are considered (Random Forest behavior).

Returns:
- the index of the best feature
- the best threshold
- the corresponding weighted Gini score
"""
function best_split(X, y; mtry=nothing)

    # X has one row per observation, so its row count must match length(y).
    size(X, 1) == length(y) ||
        error("X and y must contain the same number of observations.")

    # Number of predictor columns available at this node.
    n_features = size(X, 2)

    # Decide which features are eligible for split search.
    candidate_features =
        if isnothing(mtry)
            # Standard CART / bagging: inspect every predictor.
            1:n_features
        else
            # Random Forest: ensure the requested subset is feasible.
            mtry <= n_features ||
                error("mtry cannot exceed the number of features.")

            # randperm(n_features) returns a random permutation of feature
            # indices. Taking the first mtry elements yields a random subset
            # without replacement.
            randperm(n_features)[1:mtry]
        end

    # Track the best feature/threshold pair over all eligible features.
    best_feature = nothing
    best_threshold = nothing
    best_score = Inf

    for j in candidate_features

        # Extract feature j as a one-dimensional vector containing the values
        # for every observation currently reaching this node.
        x = X[:, j]

        # Find this feature's best threshold before comparing it against the
        # best thresholds found for the other candidate features.
        threshold, score = best_split_feature(x, y)

        # Greedily retain the feature/threshold pair with minimum impurity.
        if score < best_score
            best_score = score
            best_feature = j
            best_threshold = threshold
        end
    end

    return best_feature, best_threshold, best_score
end

end