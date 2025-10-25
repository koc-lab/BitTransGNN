#!/usr/bin/env bash

# Simple sequential runner for all quantization experiments
# Run from bert_ptq directory

set -e  # Exit on any error

# Set environment variables
export PYTHONPATH=$PYTHONPATH:../../
export GLOG_vmodule=MemcachedClient=-1

echo "Starting all quantization experiments..."
echo "Start time: $(date)"

echo "=== 8-bit osplus experiments ==="
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/osplus/20ng/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/osplus/mr/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/osplus/ohsumed/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/osplus/R8/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/osplus/R52/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/osplus/cola/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/osplus/mrpc/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/osplus/stsb/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/osplus/rte/config.yaml &&

echo "=== 8-bit twc_fine_gamma experiments ==="
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/twc_fine_gamma/20ng/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/twc_fine_gamma/mr/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/twc_fine_gamma/ohsumed/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/twc_fine_gamma/R8/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/twc_fine_gamma/R52/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/twc_fine_gamma/cola/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/twc_fine_gamma/mrpc/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/twc_fine_gamma/stsb/config.yaml &&
python ../../quant_transformer/solver/ptq_glue_quant.py --config 8-bit/twc_fine_gamma/rte/config.yaml

echo "All experiments completed!"
echo "End time: $(date)"