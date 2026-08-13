module Predict

using ..Node: TreeNode

export predict

"""
    predict(node, x)

Predict the class label for a single observation `x`
by traversing the decision tree.
"""
function predict(node::TreeNode, x)

    # Base case: leaf node
    if node.feature === nothing
        return node.prediction
    end

    # Follow the learned split rule
    if x[node.feature] <= node.threshold
        return predict(node.left, x)
    else
        return predict(node.right, x)
    end
end

end