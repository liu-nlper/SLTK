# preprocessing and train
CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/word.yml -p --train

# train only
# CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/word.yml --train

# test
CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/word.yml --test