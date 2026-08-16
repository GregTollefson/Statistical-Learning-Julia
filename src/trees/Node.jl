# The Node module defines the data structure used to represent every node in
# the decision tree. CART.jl constructs these nodes and Predict.jl traverses
# them later when making predictions.
module Node

# Export TreeNode so other modules can construct and type-annotate tree nodes.
export TreeNode

# `mutable struct` means the fields of a TreeNode could be changed after the
# object is created. In the current implementation we construct complete nodes
# and do not modify them later, but mutability keeps the structure flexible.
#
# A single TreeNode can represent either:
#   1. an internal decision node, or
#   2. a terminal leaf node.
#
# For an internal node:
#   feature   -> column index used for the split
#   threshold -> numeric split value
#   left      -> child followed when x[feature] <= threshold
#   right     -> child followed when x[feature] > threshold
#
# For a leaf node, feature/threshold/left/right are all `nothing`, and
# prediction stores the class that should be returned.
mutable struct TreeNode
    # Majority class of the observations reaching this node. At leaves this is
    # the final prediction; at internal nodes it is also retained as useful
    # node information.
    prediction

    # Integer feature index for an internal split, or `nothing` for a leaf.
    feature

    # Numeric split threshold for an internal node, or `nothing` for a leaf.
    threshold

    # Reference to the left child TreeNode, or `nothing` at a leaf.
    left

    # Reference to the right child TreeNode, or `nothing` at a leaf.
    right
end

end