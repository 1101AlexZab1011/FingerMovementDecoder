#!/bin/bash

# 4 fingers
./venv/bin/python ./LFRNNm_decoder.py -cmb "RI" "RM" "LI" "LM" --prefix EEGNet --model eegnet
./venv/bin/python ./LFRNNm_decoder.py -cmb "RI" "RM" "LI" "LM" --prefix FBCSP_ShallowNet --model fbcsp
./venv/bin/python ./LFRNNm_decoder.py -cmb "RI" "RM" "LI" "LM" --prefix Deep4 --model deep4

# left vs right
./venv/bin/python ./LFRNNm_decoder.py -cmb "RI RM" "LI LM" --prefix EEGNet --model eegnet
./venv/bin/python ./LFRNNm_decoder.py -cmb "RI RM" "LI LM" --prefix FBCSP_ShallowNet --model fbcsp
./venv/bin/python ./LFRNNm_decoder.py -cmb "RI RM" "LI LM" --prefix Deep4 --model deep4

# left
./venv/bin/python ./LFRNNm_decoder.py -cmb "LI" "LM" --prefix EEGNet --model eegnet
./venv/bin/python ./LFRNNm_decoder.py -cmb "LI" "LM" --prefix FBCSP_ShallowNet --model fbcsp
./venv/bin/python ./LFRNNm_decoder.py -cmb "LI" "LM" --prefix Deep4 --model deep4

# right
./venv/bin/python ./LFRNNm_decoder.py -cmb "RI" "RM" --prefix EEGNet --model eegnet
./venv/bin/python ./LFRNNm_decoder.py -cmb "RI" "RM" --prefix FBCSP_ShallowNet --model fbcsp
./venv/bin/python ./LFRNNm_decoder.py -cmb "RI" "RM" --prefix Deep4 --model deep4
