using MutualInformation
using Base.Test


@test 0 == entropy_estimator([1,1,1,1,1])
@test -log(1/6) ≈ entropy_estimator([1,2,3,4,5,6])

@test 0 == entropy_naive_counts([6])
@test -log(1/6) ≈ entropy_naive_counts([1,1,1,1,1,1])

# From library(entropy) in R
@test 3.8405493104064847 ≈ entropy_cs_counts([1,1,1,1,1,1])
@test 2.201137101279585 ≈ entropy_cs_counts([4,2,3,2,4,2,1,1])
@test 1.9683824087283812 ≈ entropy_naive_counts([4,2,3,2,4,2,1,1])
