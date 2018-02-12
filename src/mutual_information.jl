function mi_freqs(probs::AbstractMatrix{Float64}; normalize::Bool=false)
    ee = entropy_freqs(probs)

    ex = entropy_freqs(sum(probs, 1))
    ey = entropy_freqs(sum(probs, 2))

    mi = ex + ey - ee
    return normalize ? min(mi / min(ex, ey), 1.) : mi
end


function mutual_information{T<:Number}(x::AbstractVector{T}, y::AbstractVector{T},
                                       ex, ey; params::CountsMI=CountsMI())
  ee = joint_entropy(x, y; method=params.method)

  if params.adjusted
    ee_shuffle = joint_entropy(x, shuffle(y); method=params.method)
    mi_shuffle = ex + ey - ee_shuffle
    mi = ee_shuffle - ee
    params.normalize ? mi / (min(ex, ey) - mi_shuffle) : mi
  else
    mi = ex + ey - ee
    params.normalize ? min(mi / min(ex, ey), 1.) : mi
  end
end

function mutual_information{T<:Number}(x::AbstractVector{T}, y::AbstractVector{T},
                                       ; params::CountsMI=CountsMI())
  ex = entropy_estimator(x)
  ey = entropy_estimator(y)
  mutual_information(x, y, ex, ey; params=params)
end

function mutual_information(data::AbstractMatrix; params::CountsMI=CountsMI())
    M = size(data, 2)
    mi_sym = Array{Float64}(M, M)

    for i =1:M
      mi_sym[i,i] = entropy_estimator(@view data[:,i]; method=params.method)
    end

    for i = 1:M, j=i+1:M
        ex = mi_sym[i,i]
        ey = mi_sym[j,j]

        x = @view data[:,i]
        y = @view data[:,j]

        mi_sym[i, j] = mutual_information(x, y, ex, ey; params=params)

        # Drop null or negative values
        if mi_sym[i, j] < Base.eps()
            mi_sym[i, j] = 0.
        end

        mi_sym[j,i] = mi_sym[i,j]  # tbr
    end
    return Symmetric(mi_sym)
end

function mutual_information(p::SolidPrior; params::CountsMI=CountsMI())
  mutual_information(p.data; params=params)
end

function mutual_information(p::PairwisePrior)
  p.mi_matrix
end
