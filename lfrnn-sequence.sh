#!/bin/bash 

./venv/bin/python ./LFRNN_decoder.py -es B9 B10 B11 B12 -cmb "RI" "RM" "LI" "LM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B7 B8 B9 B10 B11 B12 -cmb "RI" "RM" "LI" "LM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B5 B6 B7 B8 B9 B10 B11 B12 -cmb "RI" "RM" "LI" "LM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 -cmb "RI" "RM" "LI" "LM" --postfix seq

./venv/bin/python ./LFRNN_decoder.py -es B9 B10 B11 B12 -cmb "RI RM" "LI LM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B7 B8 B9 B10 B11 B12 -cmb "RI RM" "LI LM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B5 B6 B7 B8 B9 B10 B11 B12 -cmb "RI RM" "LI LM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 -cmb "RI RM" "LI LM" --postfix seq

./venv/bin/python ./LFRNN_decoder.py -es B9 B10 B11 B12 -cmb "LI" "LM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B7 B8 B9 B10 B11 B12 -cmb "LI" "LM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B5 B6 B7 B8 B9 B10 B11 B12 -cmb "LI" "LM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 -cmb "LI" "LM" --postfix seq

./venv/bin/python ./LFRNN_decoder.py -es B9 B10 B11 B12 -cmb "RI" "RM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B7 B8 B9 B10 B11 B12 -cmb "RI" "RM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B5 B6 B7 B8 B9 B10 B11 B12 -cmb "RI" "RM" --postfix seq
./venv/bin/python ./LFRNN_decoder.py -es B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 -cmb "RI" "RM" --postfix seq