#!/bin/bash 

./venv/bin/python ./LFCNN_decoder.py -es B9 B10 B11 B12 -cmb "RI" "RM" "LI" "LM" --postfix B1-B8
./venv/bin/python ./LFCNN_concatenated.py -es B9 B10 B11 B12 -cmb "RI" "RM" "LI" "LM" --postfix B1-B8

./venv/bin/python ./LFCNN_decoder.py -es B9 B10 B11 B12 -cmb "RI RM" "LI LM" --postfix B1-B8
./venv/bin/python ./LFCNN_concatenated.py -es B9 B10 B11 B12 -cmb "RI RM" "LI LM" --postfix B1-B8

./venv/bin/python ./LFCNN_decoder.py -es B9 B10 B11 B12 -cmb "LI" "LM" --postfix B1-B8
./venv/bin/python ./LFCNN_concatenated.py -es B9 B10 B11 B12 -cmb "LI" "LM" --postfix B1-B8

./venv/bin/python ./LFCNN_decoder.py -es B9 B10 B11 B12 -cmb "RI" "RM" --postfix B1-B8
./venv/bin/python ./LFCNN_concatenated.py -es B9 B10 B11 B12 -cmb "RI" "RM" --postfix B1-B8