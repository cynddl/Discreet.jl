function mutual_information_contingency(
    probs::AbstractMatrix{Float64}; normalize::Bool=false)

    ee = entropy(ProbabilityWeights(probs[:]))
    ex = entropy(ProbabilityWeights(sum(probs, 1)))
    ey = entropy(ProbabilityWeights(sum(probs, 2)))

    mi = ex + ey - ee
    return normalize ? min(mi / min(ex, ey), 1.) : mi
end


function mutual_information(
    x::AbstractVector, y::AbstractVector, ex, ey; method=:Naive, adjusted=false, normalize=false)

    # Compute the joint entropy without adjustment nor normalization
    ee = joint_entropy(x, y; method=method)

    if adjusted
        ee_shuffle = joint_entropy(x, shuffle(y); method=method)
        mi_shuffle = ex + ey - ee_shuffle
        mi = ee_shuffle - ee
        normalize ? mi / (min(ex, ey) - mi_shuffle) : mi
    else
        mi = ex + ey - ee
        normalize ? min(mi / min(ex, ey), 1.) : mi
    end
end

function mutual_information(
    x::AbstractVector, y::AbstractVector;
    method=:Naive, adjusted=false, normalize=false)

    ex = entropy(x)
    ey = entropy(y)
    mutual_information(
        x, y, ex, ey; method=method, adjusted=adjusted, normalize=normalize)
end

function mutual_information(
    data::AbstractMatrix; method=:Naive, adjusted=false, normalize=false)

    M = size(data, 2)
    mi_sym = Array{Float64}(M, M)

    for i =1:M
        mi_sym[i,i] = entropy(@view data[:,i]; method=method)
    end

    for i = 1:M, j=i+1:M
        ex = mi_sym[i,i]
        ey = mi_sym[j,j]

        x = @view data[:,i]
        y = @view data[:,j]

        mi_sym[i, j] = mutual_information(
            x, y, ex, ey;
            method=method, adjusted=adjusted, normalize=normalize)

        # Drop null or negative values
        if mi_sym[i, j] < Base.eps()
            mi_sym[i, j] = 0.
        end

        mi_sym[j,i] = mi_sym[i,j]  # tbr
    end
    mi_sym
end
