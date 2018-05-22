module Discreet

using StatsBase

export entropy, estimate_entropy, estimate_joint_entropy
export mutual_information_contingency, mutual_information

include("entropy.jl")
include("mutual_information.jl")

end
