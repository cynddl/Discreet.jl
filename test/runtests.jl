using MutualInformation
import StatsBase: FrequencyWeights, ProbabilityWeights
using Base.Test


# Naive estimator
@test 0 == entropy([1,1,1,1,1,1])
@test -log(1/6) ≈ entropy(FrequencyWeights([1,1,1,1,1,1]))
@test 0 == entropy(FrequencyWeights([6]))

# From library(entropy) in R
@test 3.840549310406 ≈ entropy(FrequencyWeights([1,1,1,1,1,1]); method=:CS)
@test 2.201137101279 ≈ entropy(FrequencyWeights([4,2,3,2,4,2,1,1]); method=:CS)
@test 1.968382408728 ≈ entropy(FrequencyWeights([4,2,3,2,4,2,1,1]))
@test 2.379602895309 ≈ entropy(
    FrequencyWeights([4,2,3,0,2,4,0,0,2,1,1]);
    method=:Shrink)
