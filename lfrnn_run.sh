#!/bin/bash 

./venv/bin/python ./LFRNN_decoder.py -es B9 B10 B11 B12 -cmb "RI" "RM" "LI" "LM" --postfix B1-B8_LFRNN
./venv/bin/python ./LFRNN_concatenated.py -es B9 B10 B11 B12 -cmb "RI" "RM" "LI" "LM" --postfix B1-B8_LFRNN

./venv/bin/python ./LFRNN_decoder.py -es B9 B10 B11 B12 -cmb "RI RM" "LI LM" --postfix B1-B8_LFRNN
./venv/bin/python ./LFRNN_concatenated.py -es B9 B10 B11 B12 -cmb "RI RM" "LI LM" --postfix B1-B8_LFRNN

./venv/bin/python ./LFRNN_decoder.py -es B9 B10 B11 B12 -cmb "LI" "LM" --postfix B1-B8_LFRNN
./venv/bin/python ./LFRNN_concatenated.py -es B9 B10 B11 B12 -cmb "LI" "LM" --postfix B1-B8_LFRNN

./venv/bin/python ./LFRNN_decoder.py -es B9 B10 B11 B12 -cmb "RI" "RM" --postfix B1-B8_LFRNN
./venv/bin/python ./LFRNN_concatenated.py -es B9 B10 B11 B12 -cmb "RI" "RM" --postfix B1-B8_LFRNN