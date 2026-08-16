# Predict.jl contains the inference-time tree traversal logic. Training creates
# a linked TreeNode structure; prediction follows that structure from the root
# to a leaf for one observation at a time.
module Predict

# Import the TreeNode type so the prediction function can be type-annotated.
using ..Node: TreeNode

export predict

"""
    predict(node, x)

Predict the class label for a single observation `x`
by traversing the decision tree.
"""
function predict(node::TreeNode, x)

    # Base case for the recursion: a leaf has no feature index because there is
    # no further decision rule to evaluate. Return the class stored at the leaf.
    if node.feature === nothing
        return node.prediction
    end

    # Evaluate the split learned during training. The feature index selects one
    # component of the observation vector x, and the threshold determines which
    # child subtree should be followed.
    if x[node.feature] <= node.threshold
        # The observation satisfies the node's split rule, so recurse down the
        # left subtree.
        return predict(node.left, x)
    else
        # Otherwise recurse down the right subtree.
        return predict(node.right, x)
    end
end

end