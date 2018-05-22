function mutual_information_contingency(probs::AbstractMatrix{<:Number}; normalize::Bool=false)
    ee = entropy(ProbabilityWeights(probs[:]))
    ex = entropy(ProbabilityWeights(sum(probs, 1)[:]))
    ey = entropy(ProbabilityWeights(sum(probs, 2)[:]))

    mi = ex + ey - ee
    return normalize ? min(mi / min(ex, ey), 1.) : mi
end


function mutual_information(
    x::AbstractVector{<:Number}, y::AbstractVector{<:Number}, ex::T, ey::T; method=:Naive, adjusted=false, normalize=false) where T<:Number

    # Compute the joint entropy without adjustment nor normalization
    ee = estimate_joint_entropy(x, y; method=method)

    if adjusted
        ee_shuffle = estimate_joint_entropy(x, shuffle(y); method=method)
        mi_shuffle = ex + ey - ee_shuffle
        mi = ee_shuffle - ee
        normalize ? mi / (min(ex, ey) - mi_shuffle) : mi
    else
        mi = ex + ey - ee
        normalize ? min(mi / min(ex, ey), 1.) : mi
    end
end

function mutual_information(
    x::AbstractVector{<:Number}, y::AbstractVector{<:Number};
    method=:Naive, adjusted=false, normalize=false)

    ex = estimate_entropy(x)
    ey = estimate_entropy(y)
    mutual_information(
        x, y, ex, ey; method=method, adjusted=adjusted, normalize=normalize)
end

function mutual_information(data::AbstractMatrix{<:Number}; method::Symbol=:Naive, adjusted::Bool=false, normalize::Bool=false)
    M = size(data, 2)
    mi_sym = Array{Float64}(M, M)

    for i =1:M
        mi_sym[i,i] = estimate_entropy(@view data[:,i]; method=method)
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
