module Split

export candidate_thresholds
export best_split_feature
"""
    candidate_thresholds(x)

Generate candidate split thresholds for a feature vector.

The thresholds are the midpoints between consecutive unique,
sorted feature values.
"""
function candidate_thresholds(x)

    values = sort(unique(x))

    if length(values) < 2
        return Float64[]
    end

    thresholds = Float64[]

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

     length(x) == length(y) ||
        error("x and y must contain the same number of observations.")
    
    thresholds = candidate_thresholds(x)

    best_threshold = nothing
    best_score = Inf

    for threshold in thresholds

        left_mask = x .<= threshold
        right_mask = x .> threshold

        y_left = y[left_mask]
        y_right = y[right_mask]

        score = weighted_gini(y_left, y_right)

        if score < best_score
            best_score = score
            best_threshold = threshold
        end
    end

    return best_threshold, best_score
end

"""
    best_split(X, y)

Find the best split across all features in the feature matrix X.

Returns:
- the index of the best feature
- the best threshold
- the corresponding weighted Gini score
"""
function best_split(X, y)

    size(X, 1) == length(y) ||
        error("X and y must contain the same number of observations.")

    n_features = size(X, 2)

    best_feature = nothing
    best_threshold = nothing
    best_score = Inf

    for j in 1:n_features

        x = X[:, j]

        threshold, score = best_split_feature(x, y)

        if score < best_score
            best_score = score
            best_feature = j
            best_threshold = threshold
        end
    end

    return best_feature, best_threshold, best_score
end


end