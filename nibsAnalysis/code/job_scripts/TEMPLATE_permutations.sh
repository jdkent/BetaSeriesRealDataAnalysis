#!/bin/bash

#$ -pe smp 54
#$ -q UI
#$ -m bea
#$ -M james-kent@uiowa.edu
#$ -o /Users/jdkent/bids/derivatives/nibsAnalysis/code/out/
#$ -e /Users/jdkent/bids/derivatives/nibsAnalysis/code/err/

conda activate aim1_valid
cd /Users/jdkent/bids/derivatives/nibsAnalysis

./run_permutations.py \
    '{"task": "taskswitch", "condition": "CONDITION1"}' \
    '{"task": "taskswitch", "condition": "CONDITION2"}' \
    DIRECTORY ../mriqc/group_bold.tsv 54 1000 FNAME
