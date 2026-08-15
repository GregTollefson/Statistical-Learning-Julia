# Build an ensemble of CART trees using bootstrap aggregation (bagging).
#
# Each tree is trained on a different bootstrap sample of the original data.
# The function also records which observations were out-of-bag for each tree
# so that the forest can later be evaluated using OOB predictions.
#
# The mtry argument controls feature subsampling:
#   mtry = nothing  -> ordinary bagged CART; every feature may be considered
#   mtry < p        -> Random Forest; only mtry candidate features are
#                     considered at each split
function build_bagged_trees(X, y, n_trees::Int; max_depth=10, mtry=nothing)
    # Store one entry per tree. Each entry will contain both the fitted tree
    # and the indices of observations that were OOB for that tree.
    forest = []

    # Construct n_trees independently bootstrapped CART models.
    for b in 1:n_trees
        # Draw a bootstrap sample and simultaneously identify the observations
        # that were left out of this particular tree's training sample.
        boot_idx, oob_idx = bootstrap_indices(length(y))

        # Use the sampled row indices to construct the bootstrap training set.
        # The ':' keeps all feature columns while boot_idx selects rows.
        X_boot = X[boot_idx, :]
        y_boot = y[boot_idx]

        # Fit one CART tree to this bootstrap sample.
        #
        # When mtry=nothing this is a normal CART tree inside a bagged ensemble.
        # When mtry is an integer, CART's split search randomly restricts the
        # candidate features, which turns the ensemble into a Random Forest.
        tree = CART.build_tree(
            X_boot,
            y_boot;
            max_depth=max_depth,
            mtry=mtry
        )

        # A NamedTuple keeps the tree together with its OOB observations.
        # This bookkeeping is what later allows predict_oob() to determine
        # which trees are allowed to vote on a particular observation.
        push!(forest, (tree=tree, oob_idx=oob_idx))
    end

    return forest
end

# Combine the class predictions from several trees using majority voting.
# The class receiving the largest number of votes becomes the ensemble's
# prediction.
function majority_vote(predictions)
    # Determine which class labels actually occur among the votes.
    classes = unique(predictions)

    # Initialize the winner using the first observed class. These values are
    # replaced whenever a class with a larger vote count is encountered.
    best_class = classes[1]
    best_count = 0

    for c in classes
        # ==(c) creates a predicate that is true whenever a prediction equals c.
        # count then counts how many trees voted for class c.
        count_c = count(==(c), predictions)

        # Keep the class with the greatest number of votes seen so far.
        if count_c > best_count
            best_count = count_c
            best_class = c
        end
    end

    return best_class
end

# Predict the class of one feature vector x using every tree supplied in trees.
# This helper expects a collection of tree objects rather than the NamedTuples
# returned by build_bagged_trees().
function predict_forest(trees, x)
    # Ask each tree for an independent prediction. Julia's comprehension
    # collects the individual predictions into a vector.
    predictions = [
        Predict.predict(tree, x)
        for tree in trees
    ]

    # Aggregate the individual tree predictions into one ensemble prediction.
    return majority_vote(predictions)
end

# Produce an out-of-bag prediction for observation i.
#
# The crucial rule is that a tree may vote on observation i ONLY if i was not
# used to train that tree. This makes the prediction an internal validation
# estimate rather than a training-set prediction.
function predict_oob(forest, X, i)
    # Class labels in the Wine data are integers, so collect votes in Int[].
    predictions = Int[]

    for member in forest
        # member contains:
        #   member.tree    -> fitted CART model
        #   member.oob_idx -> observations omitted from that tree's bootstrap
        #                     training sample
        #
        # Only use the tree if observation i was genuinely out-of-bag.
        if i in member.oob_idx
            prediction = Predict.predict(
                member.tree,
                X[i, :]
            )

            push!(predictions, prediction)
        end
    end

    # With a small forest it is possible, though increasingly unlikely as the
    # number of trees grows, that an observation was included in every
    # bootstrap sample and therefore has no OOB voters.
    if isempty(predictions)
        return nothing
    end

    # The OOB prediction is the majority vote among only the eligible trees.
    return majority_vote(predictions)
end