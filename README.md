# Discreet.jl

[![Build Status](https://github.com/cynddl/Discreet.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/cynddl/Discreet.jl/actions/workflows/CI.yml)
[![Coverage status](http://codecov.io/github/cynddl/Discreet.jl/coverage.svg?branch=master)](http://codecov.io/github/cynddl/Discreet.jl?branch=master)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

**Discreet.jl** is a Julia package for estimating entropy and mutual information from discrete samples. It provides bias-corrected estimators to improve accuracy for small, finite sample sizes with a clean API based on STatsBase frequency weights.

## Installation

```julia
using Pkg
Pkg.add("Discreet")
```

## Quick Start

```julia
using Discreet
using StatsBase: FrequencyWeights

# Estimate entropy from frequency counts
counts = FrequencyWeights([10, 5, 3, 2])
entropy(counts)                    # Naive estimator
entropy(counts; method=:CS)        # Chao-Shen bias correction  
entropy(counts; method=:Shrink)    # Shrinkage estimator

# Estimate entropy directly from data
data = ["A", "B", "A", "C", "A", "B"]
estimate_entropy(data)             # Convert to counts automatically

# Mutual information between variables
x = [1, 1, 2, 2, 3, 3]
y = [1, 2, 1, 2, 1, 2] 
mutual_information(x, y)           # Raw mutual information
mutual_information(x, y; normalize=true)  # Normalized [0,1]
```

## Entropy Estimation

Discreet supports multiple entropy estimators to handle bias in finite samples:

```julia
using StatsBase: FrequencyWeights, ProbabilityWeights
using Discreet

dist = FrequencyWeights([1, 1, 1, 1, 1, 1])
entropy(dist)  # Naive method: log(6) ≈ 1.792

entropy(dist; method=:CS)  # Chao-Shen correction: ≈ 3.840

entropy(dist; method=:Shrink)  # Shrinkage correction: ≈ 1.792

dist = ProbabilityWeights([.5, .5])
entropy(dist)  # log(2) ≈ 0.693
```

Discreet can also estimate the entropy of a sample:

```julia
using Discreet

data = ["tomato", "apple", "apple", "banana", "tomato"]
estimate_entropy(data)  # == entropy(FrequencyWeights([2, 2, 1]))
```

## Estimate mutual information

Discrete provides similar routines to estimate mutual information.

```julia
using Discreet

labels_a = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
labels_b = [1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2]
mutual_information(labels_a, labels_b)  # Naive method: ≈ 0.410

mutual_information(labels_a, labels_b; method=:CS)  # Chao-Shen correction: ≈ 0.148

mutual_information(labels_a, labels_b; normalize=true)  # Normalized score (between 0 and 1): ≈ 0.382
```

## Method Comparison

| Method | Use Case | Bias |
|--------|----------|------|
| `:Naive` | Large samples, quick estimates | High for small samples |
| `:CS` | Small samples, many categories | Low (corrects underestimation) |
| `:Shrink` | Noisy data, regularization needed | Low (shrinks toward uniform) |

## API Reference

### Entropy Functions
- `entropy(probs::ProbabilityWeights)` - Direct entropy from probabilities
- `entropy(counts::FrequencyWeights; method=:Naive)` - Entropy from counts  
- `estimate_entropy(data; method=:Naive)` - Entropy from raw data

### Mutual Information Functions  
- `mutual_information(x, y; method=:Naive, normalize=false)` - Pairwise MI
- `mutual_information(data::Matrix; ...)` - All pairwise MI in dataset
- `mutual_information_contingency(probs; normalize=false)` - MI from joint distribution
