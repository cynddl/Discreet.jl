function entropy_freqs(probs::AbstractVector{Float64})
    n = length(probs)
    rv = 0.
    @fastmath @inbounds @simd for i=1:n
        if probs[i] > 0.
            @fastmath @inbounds rv -= probs[i] * log(probs[i])
        end
    end
    return rv
end

function entropy_freqs(probs::AbstractMatrix{Float64})
    return entropy_freqs(probs[:])  # use flat memory representation
end

function entropy_naive_counts(counts::AbstractVector{Int})
  return entropy_freqs(counts / sum(counts))
end

function entropy_cs_counts(counts::AbstractVector{Int})
  # Chao-Shen (2003) entropy estimator
  n = sum(counts)
  θ_ML = counts / n

  f1 = sum(counts .== 1)
  if f1 == n
    f1 = n - 1  # avoid C=0
  end

  # Estimate coverage
  C = Float64(1 - f1 / n)
  p_a = C * θ_ML
  l_a = (1 - (1 - p_a) .^ n)

  return - sum(p_a .* log.(p_a) ./ l_a)
end

" From http://www.jmlr.org/papers/volume10/hausser09a/hausser09a.pdf "
function entropy_shrink_counts(counts::Array{Int,1})
    # Normalize frequencies
    n = sum(counts)
    θ_ML = counts / n

    # Uniform distribution
    t_k = 1 / length(θ_ML)

    # Regularization parameter
    λ = (1 - sum(θ_ML .^ 2)) ./ ((n - 1) * sum((θ_ML - t_k) .^2))
    if λ == Inf
        return entropy_freqs(θ_ML)
    end

    θ_shrink = λ * t_k + (1 - λ) * θ_ML
    return entropy_freqs(θ_shrink)
end

"""
Estimate the entropy of an array using a naive (frequencies-based),
Chao-Shen, or shrinkage estimator. Chao-Shen and shrinkage estimators reduce
the bias due to the small samples and high uniqueness.
"""
function entropy_estimator{T<:Number}(data::AbstractVector{T}; method=:Naive)
  count_values = collect(values(countmap(data; alg=:dict)))
  return entropy_estimator_counts(count_values; method=method)
end

function entropy_estimator_counts(counts::AbstractVector{Int}; method=:Naive)
  if method == :Naive
    return entropy_naive_counts(counts)
  elseif method == :Shrink
    return entropy_shrink_counts(counts)
  elseif method == :CS
    return entropy_cs_counts(counts)
  end
end

function joint_entropy{T<:Number}(x::AbstractVector{T}, y::AbstractVector{T};
                                  method=:Naive)
  @assert length(x) == length(y)
  N = length(x)

  freqs = collect(values(countmap([hash(y[i], UInt64(x[i])) for i=1:N])))
  entropy_estimator_counts(freqs; method=method)
end
