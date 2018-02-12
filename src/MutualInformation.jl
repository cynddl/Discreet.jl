module MutualInformation

using StatsBase

export entropy, joint_entropy
export mi_freqs, mutual_information

include("entropy.jl")
include("mutual_information.jl")

end
