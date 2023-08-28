#!/bin/bash

exe="$1"
nbtest="$2"
shift 2
args="$@"

total=0
for ((i=1; i<=nbtest; i++))
do
  
  output=$("$exe" "$args")
  echo "$i : $output"
  total=$(echo "$total + $output" | bc -l)
done

average=$(echo "scale=3; $total / $nbtest" | bc -l)

echo "Average = $average"

