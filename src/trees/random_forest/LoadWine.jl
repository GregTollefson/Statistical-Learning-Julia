using DelimitedFiles

function load_wine(path::String)
    data = readdlm(path, ',', Float64)

    y = Int.(data[:, 1])
    X = data[:, 2:end]

    return X, y
end