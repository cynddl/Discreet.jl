entropy_freqs(probs::AbstractVector{Float64}) =
  -sum(x->(x * log(x)), probs[probs .> 0])


entropy_naive_counts(counts::AbstractVector{Int}) =
  entropy_freqs(counts ./ sum(counts))


#=
Chao-Shen (2003) entropy estimator
=#
function entropy_cs_counts(counts::AbstractVector{Int})
  n = sum(counts)
  θ_ML = counts / n

  f1 = sum(counts .== 1)
  f1 = (f1 == n) ? n - 1 : f1 # avoid C=0

  # Estimate coverage
  C = (1. - f1 / n)
  p_a = C .* θ_ML
  l_a = (1. .- (1. - p_a) .^ n)

  return -sum(p_a .* log.(p_a) ./ l_a)
end


#=
From http://www.jmlr.org/papers/volume10/hausser09a/hausser09a.pdf
=#
function entropy_shrink_counts(counts::AbstractVector{Int})
    n = sum(counts)
    θ_ML = counts / n

    # Uniform distribution
    t_k = 1. / length(θ_ML)

    # Regularization parameter
    λ = (1. - sum(θ_ML .^ 2)) ./ ((n - 1) * sum((θ_ML - t_k) .^2))

    isinf(λ) ?
      entropy_freqs(θ_ML) :
      entropy_freqs( λ * t_k .+ (1. - λ) .* θ_ML)
end


#=
Estimate the entropy of an array using a naive (frequencies-based),
Chao-Shen, or shrinkage estimator. Chao-Shen and shrinkage estimators reduce
the bias due to the small samples and high uniqueness.
=#
function entropy_estimator(data::AbstractVector{T};
                           method=:Naive) where T <: Real

  count_values = collect(values(countmap(data)))
  return entropy_estimator_counts(count_values; method=method)
end


function entropy_estimator_counts(counts::AbstractVector{Int};
                                  method=:Naive)

  return (method == :Naive) ?
            entropy_naive_counts(counts) :
         (method == :Shrink) ?
            entropy_shrink_counts(counts) :
         (method == :CS) ?
            entropy_cs_counts(counts) :
            throw(ArgumentError("Unknown method"))
end


function joint_entropy(x::AbstractVector{T}, y::AbstractVector{T};
                       method=:Naive) where T <: Real

  @assert length(x) == length(y) "Vectors must be the same length"

  N = length(x)

  freqs = collect(values(countmap([hash(y[i], hash(x[i])) for i=1:N])))
  entropy_estimator_counts(freqs; method=method)
end
