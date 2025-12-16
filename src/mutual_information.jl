"""
    mutual_information_contingency(probs::AbstractMatrix{Float64}; normalize::Bool=false) -> Float64

Compute mutual information from a joint probability matrix (contingency table).

# Arguments
- `probs::AbstractMatrix{Float64}`: Joint probability matrix where `probs[i,j] = P(X=i, Y=j)`.
- `normalize::Bool=false`: If `true`, return normalized mutual information (0 ≤ NMI ≤ 1).

# Returns
- `Float64`: Mutual information I(X;Y), optionally normalized.

# Formula
- MI: I(X;Y) = H(X) + H(Y) - H(X,Y)
- Normalized: NMI = I(X;Y) / min(H(X), H(Y))

# Example
```julia
# Create joint probability matrix
probs = [0.25 0.25; 0.125 0.375]  # Must sum to 1
mi = mutual_information_contingency(probs)  # Raw mutual information
nmi = mutual_information_contingency(probs; normalize=true)  # Normalized
```
"""
function mutual_information_contingency(
    probs::AbstractMatrix{Float64};
    normalize::Bool = false,
)::Float64
    ee = entropy(ProbabilityWeights(probs[:]))
    ex = entropy(ProbabilityWeights(sum(probs; dims = 1)[:]))
    ey = entropy(ProbabilityWeights(sum(probs; dims = 2)[:]))

    mi = ex + ey - ee
    return normalize ? min(mi / min(ex, ey), 1.0) : mi
end

"""
    mutual_information(x::AbstractVector, y::AbstractVector, ex::Float64, ey::Float64; method=:Naive, adjusted=false, normalize=false) -> Float64

Compute mutual information between two discrete variables with precomputed marginal entropies.

# Arguments
- `x::AbstractVector`: Observations from the first variable.
- `y::AbstractVector`: Observations from the second variable.
- `ex::Float64`: Precomputed entropy of `x`.
- `ey::Float64`: Precomputed entropy of `y`.
- `method::Symbol=:Naive`: Joint entropy estimation method (`:Naive`, `:CS`, `:Shrink`).
- `adjusted::Bool=false`: If `true`, compute adjusted mutual information (AMI).
- `normalize::Bool=false`: If `true`, return normalized mutual information.

# Returns
- `Float64`: Mutual information estimate, optionally adjusted and/or normalized.

# Note
This version is useful when marginal entropies are already computed or when
performing multiple MI calculations with the same marginals.
"""
function mutual_information(
    x::AbstractVector,
    y::AbstractVector,
    ex::Float64,
    ey::Float64;
    method::Symbol = :Naive,
    adjusted::Bool = false,
    normalize::Bool = false,
)::Float64
    length(x) == length(y) || throw(
        ArgumentError(
            "Input vectors must have the same length: got $(length(x)) and $(length(y))",
        ),
    )

    # Compute the joint entropy without adjustment nor normalization
    ee = estimate_joint_entropy(x, y; method = method)

    if adjusted
        ee_shuffle = estimate_joint_entropy(x, shuffle(y); method = method)
        mi_shuffle = ex + ey - ee_shuffle
        mi = ee_shuffle - ee
        normalize ? mi / (min(ex, ey) - mi_shuffle) : mi
    else
        mi = ex + ey - ee
        normalize ? min(mi / min(ex, ey), 1.0) : mi
    end
end

"""
    mutual_information(x::AbstractVector, y::AbstractVector; method=:Naive, adjusted=false, normalize=false) -> Float64

Compute mutual information between two discrete random variables.

# Arguments
- `x::AbstractVector`: Observations from the first variable.
- `y::AbstractVector`: Observations from the second variable (must have same length as `x`).
- `method::Symbol=:Naive`: Entropy estimation method (`:Naive`, `:CS`, `:Shrink`).
- `adjusted::Bool=false`: If `true`, compute adjusted mutual information (AMI) that corrects for chance.
- `normalize::Bool=false`: If `true`, return normalized mutual information (0 ≤ NMI ≤ 1).

# Returns
- `Float64`: Mutual information estimate I(X;Y).

# Examples
```julia
labels_a = [1, 1, 1, 2, 2, 2, 3, 3, 3]
labels_b = [1, 1, 2, 2, 2, 3, 3, 3, 1]

# Basic mutual information
mi = mutual_information(labels_a, labels_b)  # ≈ 0.462

# Bias-corrected estimate
mi_cs = mutual_information(labels_a, labels_b; method=:CS)

# Normalized mutual information (0 ≤ NMI ≤ 1)
nmi = mutual_information(labels_a, labels_b; normalize=true)
```

# Formula
I(X;Y) = H(X) + H(Y) - H(X,Y)

where H denotes entropy estimated using the specified method.
"""
function mutual_information(
    x::AbstractVector,
    y::AbstractVector;
    method::Symbol = :Naive,
    adjusted::Bool = false,
    normalize::Bool = false,
)::Float64
    length(x) == length(y) || throw(
        ArgumentError(
            "Input vectors must have the same length: got $(length(x)) and $(length(y))",
        ),
    )

    ex = estimate_entropy(x)
    ey = estimate_entropy(y)
    return mutual_information(
        x,
        y,
        ex,
        ey;
        method = method,
        adjusted = adjusted,
        normalize = normalize,
    )
end

"""
    mutual_information(data::AbstractMatrix; method=:Naive, adjusted=false, normalize=false) -> Matrix{Float64}

Compute pairwise mutual information matrix for all variable pairs in a dataset.

# Arguments
- `data::AbstractMatrix`: Data matrix where each column represents a variable.
- `method::Symbol=:Naive`: Entropy estimation method (`:Naive`, `:CS`, `:Shrink`).
- `adjusted::Bool=false`: If `true`, compute adjusted mutual information.
- `normalize::Bool=false`: If `true`, return normalized mutual information matrix.

# Returns
- `Matrix{Float64}`: Symmetric matrix where `result[i,j]` is the mutual information between variables i and j. Diagonal elements contain the entropy of each variable.

# Example
```julia
# Data with 3 variables, 100 observations each
data = hcat(rand(1:5, 100), rand(1:3, 100), rand(1:4, 100))
mi_matrix = mutual_information(data)
```
In this case, we have `mi_matrix[i,i] = H(Xi)`` (entropy of variable i) and `mi_matrix[i,j] = I(Xi;Xj)` for i ≠ j (mutual information).

# Note
This function is efficient for computing all pairwise relationships in a dataset,
as it reuses computed marginal entropies and ensures the result matrix is symmetric.
"""
function mutual_information(
    data::AbstractMatrix;
    method::Symbol = :Naive,
    adjusted::Bool = false,
    normalize::Bool = false,
)::Matrix{Float64}
    M = size(data, 2)
    mi_sym = Array{Float64,2}(undef, M, M)

    # Compute diagonal elements (marginal entropies)
    @inbounds for i in 1:M
        mi_sym[i, i] = estimate_entropy(@view data[:, i]; method = method)
    end

    # Compute off-diagonal elements (pairwise MI)  
    @inbounds for i in 1:M, j in (i+1):M
        ex = mi_sym[i, i]
        ey = mi_sym[j, j]

        x = @view data[:, i]
        y = @view data[:, j]

        mi_val = mutual_information(
            x,
            y,
            ex,
            ey;
            method = method,
            adjusted = adjusted,
            normalize = normalize,
        )

        # Drop null or negative values for numerical stability
        mi_val = max(mi_val, 0.0)

        mi_sym[i, j] = mi_val
        mi_sym[j, i] = mi_val  # symmetric matrix
    end
    return mi_sym
end
