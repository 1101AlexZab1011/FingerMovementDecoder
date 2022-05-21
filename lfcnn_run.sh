#!/bin/bash

./venv/bin/python ./LFCNN_decoder.py -cmb "RI" "RM" "LI" "LM"
./venv/bin/python ./LFCNN_concatenated.py -cmb "RI" "RM" "LI" "LM"

./venv/bin/python ./LFCNN_decoder.py -cmb "RI RM" "LI LM"
./venv/bin/python ./LFCNN_concatenated.py -cmb "RI RM" "LI LM"

./venv/bin/python ./LFCNN_decoder.py -cmb "LI" "LM"
./venv/bin/python ./LFCNN_concatenated.py -cmb "LI" "LM"

./venv/bin/python ./LFCNN_decoder.py -cmb "RI" "RM"
./venv/bin/python ./LFCNN_concatenated.py -cmb "RI" "RM"