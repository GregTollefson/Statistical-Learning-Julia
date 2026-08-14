function bootstrap_indices(n::Int)
    boot_idx = rand(1:n, n)
    oob_idx = setdiff(1:n, boot_idx)

    return boot_idx, oob_idx
end