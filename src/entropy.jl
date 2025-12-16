import StatsBase: entropy, FrequencyWeights, ProbabilityWeights

"""
    entropy(probs::ProbabilityWeights) -> Float64

Compute the Shannon entropy of a probability distribution.

# Arguments
- `probs::ProbabilityWeights`: A probability distribution where weights sum to 1.

# Returns
- `Float64`: The Shannon entropy in natural units (nats).

# Formula
H(X) = -∑ p(x) log(p(x))

# Example
```julia
using StatsBase: ProbabilityWeights
probs = ProbabilityWeights([0.5, 0.5])
entropy(probs)  # ≈ 0.693 (log(2))
```
"""
function entropy(probs::ProbabilityWeights)::Float64
    result = 0.0
    @inbounds for p in probs
        if p > 0
            result -= p * log(p)
        end
    end
    return result
end

"""
    entropy(counts::FrequencyWeights; method=:Naive) -> Float64

Estimate entropy from frequency counts using various bias-correction methods.

# Arguments
- `counts::FrequencyWeights`: Frequency counts for each outcome.
- `method::Symbol`: Estimation method (`:Naive`, `:CS`, or `:Shrink`).

# Methods
- `:Naive`: Maximum likelihood estimator (no bias correction)
- `:CS`: Chao-Shen estimator (reduces bias for undersampled distributions)
- `:Shrink`: Shrinkage estimator (reduces bias via regularization)

# Returns
- `Float64`: Estimated entropy in natural units (nats).

# Examples
```julia
using StatsBase: FrequencyWeights
counts = FrequencyWeights([1, 1, 1, 1, 1, 1])

# Naive estimator
entropy(counts)  # ≈ 1.792 (log(6))

# Bias-corrected estimators
entropy(counts; method=:CS)      # ≈ 3.840
entropy(counts; method=:Shrink)  # ≈ 1.792
```
"""
function entropy(counts::FrequencyWeights; method::Symbol = :Naive)::Float64
    method == :Naive && return entropy_naive(counts)
    method == :Shrink && return entropy_shrinkage(counts)
    method == :CS && return entropy_cs(counts)
    throw(
        ArgumentError(
            "Unknown entropy estimation method: $method. Supported methods are :Naive, :CS, :Shrink",
        ),
    )
end

entropy_naive(counts::FrequencyWeights)::Float64 =
    entropy(ProbabilityWeights(counts / sum(counts)))

"""
    entropy_cs(counts::FrequencyWeights) -> Float64

Chao-Shen (2003) entropy estimator with bias correction for undersampled distributions.

This estimator adjusts for the presence of unseen species/outcomes in finite samples
by estimating sample coverage and adjusting probabilities accordingly.

# Arguments
- `counts::FrequencyWeights`: Frequency counts for each observed outcome.

# Returns
- `Float64`: Bias-corrected entropy estimate.

# Reference
Chao, A. and Shen, T.J., 2003. Nonparametric estimation of Shannon’s index of diversity when there are unseen species in sample. Environmental and Ecological Statistics, 10(4), pp.429-443.
"""
function entropy_cs(counts::FrequencyWeights)::Float64
    n = sum(counts)
    θ_ML = counts / n

    f1 = sum(counts .== 1)
    f1 = (f1 == n) ? n - 1 : f1  # avoid C=0

    # Estimate coverage
    C = (1 - f1 / n)
    p_a = C * θ_ML
    l_a = (1 .- (1 .- p_a) .^ n)

    return -sum(p_a .* log.(p_a) ./ l_a)
end

"""
    entropy_shrinkage(counts::FrequencyWeights) -> Float64

Shrinkage entropy estimator that reduces bias through regularization.

This estimator shrinks the maximum likelihood estimate toward a uniform distribution
using a regularization parameter λ that balances bias and variance.

# Arguments
- `counts::FrequencyWeights`: Frequency counts for each observed outcome.

# Returns
- `Float64`: Shrinkage-corrected entropy estimate.

# Reference
Hausser, J. and Strimmer, K., 2009. Entropy inference and the James-Stein estimator, with application to nonlinear gene association networks. Journal of Machine Learning Research, 10(7)."""
function entropy_shrinkage(counts::FrequencyWeights)::Float64
    n = sum(counts)
    θ_ML = counts / n

    # Uniform distribution
    t_k = 1 / length(θ_ML)

    den = (n - 1) * sum((θ_ML .- t_k) .^ 2)
    if den < 1e-10
        return entropy(ProbabilityWeights(θ_ML))
    else
        # Regularization parameter
        λ = (1 - sum(θ_ML .^ 2)) / den
        return entropy(ProbabilityWeights(λ * t_k .+ (1 - λ) * θ_ML))
    end
end

"""
    estimate_entropy(data::AbstractVector; method::Symbol=:Naive) -> Float64

Estimate entropy directly from data samples using various bias-correction methods.

# Arguments
- `data::AbstractVector`: Vector of discrete observations (any comparable type).
- `method::Symbol`: Estimation method (`:Naive`, `:CS`, or `:Shrink`).

# Methods
- `:Naive`: Maximum likelihood estimator (no bias correction)
- `:CS`: Chao-Shen estimator (reduces bias for undersampled distributions)
- `:Shrink`: Shrinkage estimator (reduces bias via regularization)

# Returns
- `Float64`: Estimated entropy in natural units (nats).

# Example
```julia
data = ["apple", "banana", "apple", "cherry", "banana", "apple"]
estimate_entropy(data)              # ≈ 1.011 (naive estimate)
estimate_entropy(data; method=:CS)  # ≈ 1.026 (bias-corrected estimate)
```

# Note
This function first computes frequency counts from the data, then applies
the specified entropy estimation method. For repeated calculations with the
same data, consider computing frequencies once and using `entropy()` directly.
"""
function estimate_entropy(data::AbstractVector; method::Symbol = :Naive)::Float64
    counts = countmap(data)
    freqs = FrequencyWeights(collect(values(counts)))
    return entropy(freqs; method = method)
end

"""
    estimate_joint_entropy(x::AbstractVector, y::AbstractVector; method::Symbol=:Naive) -> Float64

Estimate the joint entropy H(X,Y) of two discrete random variables.

# Arguments
- `x::AbstractVector`: Observations from the first variable.
- `y::AbstractVector`: Observations from the second variable (must have same length as `x`).
- `method::Symbol`: Estimation method (`:Naive`, `:CS`, or `:Shrink`).

# Returns
- `Float64`: Estimated joint entropy in natural units (nats).

# Example
```julia
x = [1, 1, 2, 2, 3, 3]
y = ["a", "b", "a", "c", "b", "c"]
estimate_joint_entropy(x, y)  # ≈ 1.792 (naive estimate)
```

# Note
Joint entropy is computed by treating each unique (x,y) pair as a single outcome
and computing the entropy of the resulting distribution.
"""
function estimate_joint_entropy(
    x::AbstractVector,
    y::AbstractVector;
    method::Symbol = :Naive,
)::Float64
    length(x) == length(y) || throw(
        ArgumentError(
            "Input vectors must have the same length: got $(length(x)) and $(length(y))",
        ),
    )

    count = countmap([hash(yᵢ, hash(xᵢ)) for (xᵢ, yᵢ) in zip(x, y)]; alg = :dict)
    freqs = FrequencyWeights(collect(values(count)))
    return entropy(freqs; method = method)
end
