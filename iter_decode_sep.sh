#!/bin/bash

# LFCNN

# 4 cases
for i in {0..10}
    do 
        ./venv/bin/python ./LFCNN_separated.py -cms "B1 B2 B3" "B10 B11 B12" -cmc "LI" "LM" "RI" "RM" --no-params --tmin 0
    done

# left vs right
for i in {0..10}
    do 
        ./venv/bin/python ./LFCNN_separated.py -cms "B1 B2 B3" "B10 B11 B12" -cmc "LI LM" "RI RM" --no-params --tmin 0
    done

# left
for i in {0..10}
    do 
        ./venv/bin/python ./LFCNN_separated.py -cms "B1 B2 B3" "B10 B11 B12" -cmc "LI" "LM" --no-params --tmin 0
    done

# right
for i in {0..10}
    do 
        ./venv/bin/python ./LFCNN_separated.py -cms "B1 B2 B3" "B10 B11 B12" -cmc "RI" "RM" --no-params --tmin 0
    done


# # LFRNN

# # 4 cases
# for i in {0..10}
#     do 
#         ./venv/bin/python ./LFCNN_separated.py -cms "B1 B2" "B7 B8" -cmc "LI" "LM" "RI" "RM" --no-params -m LFRNN --postfix LFRNN
#     done

# left vs right
# for i in {0..10}
#     do 
#         ./venv/bin/python ./LFCNN_separated.py -cms "B1 B2" "B7 B8" -cmc "LI LM" "RI RM" --no-params -m LFRNN --postfix LFRNN
#     done

# # left
# for i in {0..10}
#     do 
#         ./venv/bin/python ./LFCNN_separated.py -cms "B1 B2" "B7 B8" -cmc "LI" "LM" --no-params -m LFRNN --postfix LFRNN
#     done

# # right
# for i in {0..10}
#     do 
#         ./venv/bin/python ./LFCNN_separated.py -cms "B1 B2" "B7 B8" -cmc "RI" "RM" --no-params -m LFRNN --postfix LFRNN
#     done


# # seq
# for i in {0..10}
#     do 
#         ./lfcnn-sequence.sh
#     done

# for i in {0..10}
#     do 
#         ./lfrnn-sequence.sh
#     done