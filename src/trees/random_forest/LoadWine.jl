# DelimitedFiles is part of Julia's standard library and provides readdlm(),
# which is sufficient for this simple comma-delimited numeric dataset.
using DelimitedFiles

# Load the UCI Wine dataset and separate the response variable from features.
#
# The raw Wine file is arranged as:
#     class, feature_1, feature_2, ..., feature_13
#
# We return data in the conventional machine-learning form:
#     X -> n x p feature matrix
#     y -> length-n vector of class labels
function load_wine(path::String)
    # Read every value as Float64. The source file contains only numeric data,
    # so no header parsing or missing-value handling is required here.
    data = readdlm(path, ',', Float64)

    # Column 1 contains the class label (1, 2, or 3). readdlm read those labels
    # as Float64 along with the rest of the matrix, so explicitly convert them
    # back to Int for classification.
    y = Int.(data[:, 1])

    # Columns 2 through the final column are the 13 predictor variables.
    X = data[:, 2:end]

    return X, y
end