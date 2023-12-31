#!/bin/bash

test_type="benchmark_covariance_fit"
repetitions=30

for model_name in "VexprHandsOnGP" "VexprPartialHandsOnGP"; do
  for compile in "" "--compile"; do
    for num_models in {1..20}; do
      for vectorize in "" "--vectorize"; do
        cmd="./profile_performance_test.sh --model-name $model_name --test $test_type --num-models $num_models --repetitions $repetitions $vectorize $compile"
        echo "Running: $cmd"
        eval $cmd
      done
    done
  done
done
