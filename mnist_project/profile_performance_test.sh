#!/bin/bash
set -e

# Run benchmark
command="nsys profile -o my_report --force-overwrite true --trace=cuda,nvtx python run_performance_test.py $@"
echo "Executing command: $command"
$command

# Print stats
nsys export --type sqlite --output my_report.sqlite my_report.nsys-rep --force-overwrite true
python print_nsys_stats.py my_report.sqlite
