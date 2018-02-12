module MutualInformation

using StatsBase

export entropy_freqs, entropy_naive_counts, entropy_cs_counts,
       entropy_shrink_counts
export entropy_estimator, entropy_estimator_counts, joint_entropy

export mi_freqs, mutual_information

include("entropy.jl")
include("mutual_information.jl")

end
