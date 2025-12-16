using Discreet
import StatsBase: FrequencyWeights, ProbabilityWeights
using Random

using Test

freqs_uniform = FrequencyWeights([1, 1, 1, 1, 1, 1])
probs_uniform = ProbabilityWeights([0.5, 0.5])
sample = [:a, :b, :c, :d, :e, :f]

@testset "Entropy routines" begin
    ## Naive estimator
    @test log(6) ≈ entropy(freqs_uniform)
    @test 0 == entropy(FrequencyWeights([6]))
    @test log(6) ≈ estimate_entropy(sample)

    ## CS and shrinkage estimates, from library(entropy) in R
    @test 3.840549310406 ≈ entropy(FrequencyWeights([1, 1, 1, 1, 1, 1]); method = :CS)
    @test 2.201137101279 ≈ entropy(FrequencyWeights([4, 2, 3, 2, 4, 2, 1, 1]); method = :CS)
    @test 1.968382408728 ≈ entropy(FrequencyWeights([4, 2, 3, 2, 4, 2, 1, 1]))
    @test 2.379602895309 ≈
          entropy(FrequencyWeights([4, 2, 3, 0, 2, 4, 0, 0, 2, 1, 1]); method = :Shrink)

    ## Any other method
    @test_throws ArgumentError entropy(freqs_uniform; method = :NotImplemented)
    @test_throws ArgumentError estimate_entropy(sample; method = :NotImplemented)

    ## Joint entropy
    @test log(6) ≈ estimate_joint_entropy(sample, sample)
    @test 3.840549310406 ≈ estimate_joint_entropy(sample, sample; method = :CS)
end

@testset "Mutual information routines" begin
    labels_a = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    labels_b = [1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2]

    mi = mutual_information(labels_a, labels_b)
    @test 0.41022 ≈ mi atol=1e-5

    e_a = estimate_entropy(labels_a)
    e_b = estimate_entropy(labels_b)
    mi_normalized = mutual_information(labels_a, labels_b; normalize = true)
    @test mi / min(e_a, e_b) ≈ mi_normalized

    data = hcat(sample, sample, sample)
    @test log(6) * ones(3, 3) ≈ mutual_information(data)
end

@testset "Contingency tables" begin
    d = rand(10, 10)
    table = d / sum(d)
    mi_1 = mutual_information_contingency(table)
    e_a, e_b = entropy(sum(table, dims = 1)), entropy(sum(table, dims = 2))
    mi_2 = e_a + e_b - entropy(table[:])
    @test mi_1 ≈ mi_2

    ## Normalized MI
    mi_norm = mutual_information_contingency(table; normalize = true)
    @test mi_norm ≈ mi_1 / min(e_a, e_b)

    ## Table with no association
    @test 0 ≈ mutual_information_contingency(ones(20, 5) / 100) atol=1e-10
end

@testset "Edge cases and error handling" begin
    ## Empty and single element cases  
    @test entropy(FrequencyWeights(Int[])) == 0.0

    ## Single outcome (zero entropy)
    single_outcome = FrequencyWeights([10])
    @test entropy(single_outcome) == 0.0
    @test entropy(single_outcome; method = :CS) == 0.0
    @test entropy(single_outcome; method = :Shrink) == 0.0

    ## Single element data
    @test estimate_entropy([1]) == 0.0
    @test estimate_entropy(["same"]) == 0.0

    ## Mismatched vector lengths
    x_short = [1, 2]
    y_long = [1, 2, 3, 4]
    @test_throws ArgumentError mutual_information(x_short, y_long)
    @test_throws ArgumentError estimate_joint_entropy(x_short, y_long)

    ## Invalid method names
    counts = FrequencyWeights([1, 2, 3])
    @test_throws ArgumentError entropy(counts; method = :InvalidMethod)
    @test_throws ArgumentError estimate_entropy([1, 2, 3]; method = :InvalidMethod)

    ## Large frequency values (numerical stability)
    large_counts = FrequencyWeights([1000, 2000, 3000])
    @test entropy(large_counts) > 0
    @test entropy(large_counts; method = :CS) > 0
    @test entropy(large_counts; method = :Shrink) > 0
end

@testset "Normalized mutual information bounds" begin
    ## Perfect correlation (MI should equal min entropy)
    x = [1, 1, 2, 2, 3, 3]
    y = x  # Perfect correlation
    mi = mutual_information(x, y)
    nmi = mutual_information(x, y; normalize = true)
    @test nmi ≈ 1.0

    ## Independent variables (MI should be close to 0)
    Random.seed!(123)
    x_indep = rand(1:5, Int(1e7))
    y_indep = rand(1:5, Int(1e7))
    mi_indep = mutual_information(x_indep, y_indep)
    nmi_indep = mutual_information(x_indep, y_indep; normalize = true)
    @test 0 < mi_indep < 1e-6
    @test 0 < nmi_indep < 1e-6
end

@testset "Matrix mutual information consistency" begin
    ## Test pairwise consistency
    data = hcat([1, 1, 2, 2, 3, 3], [1, 2, 1, 2, 1, 2], [3, 3, 3, 1, 1, 1])
    mi_matrix = mutual_information(data)

    ## Matrix should be symmetric
    @test mi_matrix ≈ mi_matrix'

    ## Diagonal should contain entropies
    for i in 1:size(data, 2)
        expected_entropy = estimate_entropy(data[:, i])
        @test mi_matrix[i, i] ≈ expected_entropy
    end

    ## Off-diagonal elements should match pairwise calculations
    for i in 1:size(data, 2), j in (i+1):size(data, 2)
        expected_mi = mutual_information(data[:, i], data[:, j])
        @test mi_matrix[i, j] ≈ expected_mi
    end
end

@testset "Entropy method comparisons" begin
    ## CS estimator should be higher for small samples (corrects underestimation)
    small_sample_counts = FrequencyWeights([1, 1, 1, 1, 1, 1])
    h_naive = entropy(small_sample_counts; method = :Naive)
    h_cs = entropy(small_sample_counts; method = :CS)
    @test h_cs > h_naive

    ## For large samples, methods should converge
    large_sample_counts = FrequencyWeights([100, 100, 100, 100])
    h_naive_large = entropy(large_sample_counts; method = :Naive)
    h_cs_large = entropy(large_sample_counts; method = :CS)
    h_shrink_large = entropy(large_sample_counts; method = :Shrink)
    @test abs(h_naive_large - h_cs_large) < 0.1
    @test abs(h_naive_large - h_shrink_large) < 0.1
end
