# Generate the row indices used for one bootstrap sample.
#
# A bootstrap sample contains n draws from the original n observations,
# sampled WITH replacement. Because sampling is with replacement, some
# observations may appear multiple times while others may not appear at all.
#
# The observations that are not selected for a particular bootstrap sample
# are called the out-of-bag (OOB) observations. They give us a convenient
# validation set for that tree without requiring a separate train/test split.
function bootstrap_indices(n::Int)
    # Draw n integers independently from 1:n.
    #
    # Example for n = 5:
    #     boot_idx = [2, 5, 2, 1, 5]
    #
    # Rows 2 and 5 are repeated because sampling is with replacement.
    boot_idx = rand(1:n, n)

    # Find the observations that never appeared in boot_idx.
    #
    # Continuing the example above, rows 3 and 4 were never selected, so:
    #     oob_idx = [3, 4]
    #
    # setdiff treats boot_idx as a collection of selected row numbers; repeated
    # values do not matter for determining which original rows were omitted.
    oob_idx = setdiff(1:n, boot_idx)

    # Return both sets of indices because tree construction uses boot_idx,
    # while later OOB evaluation uses oob_idx.
    return boot_idx, oob_idx
end