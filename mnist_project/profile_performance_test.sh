#!/bin/bash
set -e

# Default values
SWEEP_NAME="mnist1"
MODEL_NAME="VexprPartialHandsOnGP"
TEST="benchmark_covariance_fit"
REPETITIONS=30
COMPILE=false
VECTORIZE=false
NUM_MODELS=1

# Parse input args
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --sweep-name)
        SWEEP_NAME="$2"
        shift
        shift
        ;;
        --model-name)
        MODEL_NAME="$2"
        shift
        shift
        ;;
        --test)
        TEST="$2"
        shift
        shift
        ;;
        --repetitions)
        REPETITIONS="$2"
        shift
        shift
        ;;
        --compile)
        COMPILE=true
        shift
        ;;
        --vectorize)
        VECTORIZE=true
        shift
        ;;
        --num-models)
        NUM_MODELS="$2"
        shift
        shift
        ;;
        *)    # unknown option
        shift # past argument
        ;;
    esac
done

echo "SWEEP_NAME: $SWEEP_NAME"
echo "MODEL_NAME: $MODEL_NAME"
echo "TEST: $TEST"
echo "REPETITIONS: $REPETITIONS"
echo "COMPILE: $COMPILE"
echo "VECTORIZE: $VECTORIZE"
echo "NUM_MODELS: $NUM_MODELS"

output_dir="results/perf"
mkdir -p $output_dir

filename="$output_dir/$SWEEP_NAME-$MODEL_NAME-$TEST-repetitions$REPETITIONS-compile$COMPILE-vectorize$VECTORIZE-num_models$NUM_MODELS.csv"

# Rebuild the args list using the new values
args=""
args+=" --sweep-name $SWEEP_NAME"
args+=" --model-name $MODEL_NAME"
args+=" --test $TEST"
args+=" --repetitions $REPETITIONS"
if [ "$COMPILE" = true ] ; then
    args+=" --compile"
fi
if [ "$VECTORIZE" = true ] ; then
    args+=" --vectorize"
fi
args+=" --num-models $NUM_MODELS"

# Run benchmark
command="nsys profile -o my_report --force-overwrite true --trace=cuda,nvtx python run_performance_test.py $args"
echo "Executing command: $command"
profile_output=$($command)

# Get stats
nsys export --type sqlite --output my_report.sqlite my_report.nsys-rep --force-overwrite true
stats_output=$(python print_nsys_stats.py my_report.sqlite)

duration=$(echo "$stats_output" | grep "Duration:" | awk '{print $2}')
cpu_wait_time=$(echo "$stats_output" | grep "Estimated CPU wait time:" | awk '{print $5}')
gpu_active_time=$(echo "$stats_output" | grep "Estimated GPU active time:" | awk '{print $5}')

# Write stats to csv file
echo "Writing stats to $filename"
echo "duration,cpu_wait_time,gpu_active_time" > $filename
echo "$duration,$cpu_wait_time,$gpu_active_time" >> $filename