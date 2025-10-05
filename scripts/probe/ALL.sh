#!/bin/bash
# Author: Shijie Xu
# Date: 2024-12-23
# Description: Run all the scripts to collect probes for all metals
# "ZN" "CA" "MG" "MN" "FE" "CU" "FE2" "CO" "NA" "CU1" "K" "NI" "CD" "MN3" "AG" "3CO" "HG" "PB"
metals=("ZN" "CA" "MG" "MN" "FE" "CU" "FE2" "CO" "NA" "CU1" "K" "NI" "CD" "MN3" "HG")
for metal in "${metals[@]}"; do
    python scripts/probe/3.collect_probes.py --metal "$metal"
done