module Metrics

export confusion_matrix

"""
    confusion_matrix(y_true, y_pred)

Compute a confusion matrix from true and predicted class labels.

Rows represent actual classes.
Columns represent predicted classes.
"""
function confusion_matrix(y_true, y_pred)

    classes = unique(vcat(y_true, y_pred))
    n_classes = length(classes)

    cm = zeros(Int, n_classes, n_classes)

    for i in eachindex(y_true)

        actual = findfirst(==(y_true[i]), classes)
        predicted = findfirst(==(y_pred[i]), classes)

        cm[actual, predicted] += 1
    end

    return cm, classes
end

end