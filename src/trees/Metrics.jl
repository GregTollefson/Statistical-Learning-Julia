# Metrics.jl contains evaluation helpers that are independent of the tree
# construction itself. Keeping metrics separate makes it easier to reuse the
# same evaluation code for CART, bagging, Random Forest, and later models.
module Metrics

export confusion_matrix

"""
    confusion_matrix(y_true, y_pred)

Compute a confusion matrix from true and predicted class labels.

Rows represent actual classes.
Columns represent predicted classes.
"""
function confusion_matrix(y_true, y_pred)

    # Combine the true and predicted labels before calling unique() so that the
    # matrix includes any class appearing in either vector. This protects
    # against the case where a model predicts a class absent from y_true, or
    # fails to predict one of the classes that is present.
    classes = unique(vcat(y_true, y_pred))
    n_classes = length(classes)

    # Allocate a square integer matrix initialized to zero. Entry (i, j) will
    # count observations whose actual class is classes[i] and whose predicted
    # class is classes[j].
    cm = zeros(Int, n_classes, n_classes)

    # eachindex(y_true) returns valid indices for the label vector and is the
    # idiomatic Julia choice when the exact indexing scheme need not be assumed.
    for i in eachindex(y_true)

        # Find the row associated with the observation's actual class.
        # ==(value) constructs a predicate that findfirst() applies to classes.
        actual = findfirst(==(y_true[i]), classes)

        # Find the column associated with the model's predicted class.
        predicted = findfirst(==(y_pred[i]), classes)

        # Increment the appropriate confusion-matrix cell for this observation.
        cm[actual, predicted] += 1
    end

    # Return the class ordering as well as the matrix because row/column indices
    # are only meaningful when we know which class each index represents.
    return cm, classes
end

end