#Options:
#  -h, --help            show this help message and exit
#  --ri=ROOT_IDX, --root_idx=ROOT_IDX
#                        数据索引根目录
#  --rv=ROOT_VOC, --root_voc=ROOT_VOC
#                        字典根目录
#  --pm=PATH_MODEL, --path_model=PATH_MODEL
#                        模型路径
#  --bs=BATCH_SIZE, --batch_size=BATCH_SIZE
#                        batch size
#  -g, --cuda            是否使用GPU加速
#  --nw=NB_WORK          加载数据的线程数
#  -o OUTPUT, --output=OUTPUT
#                        预测结果存放路径

CUDA_VISIBLE_DEVICES=2 python3 ../test.py \
    --ri ../data/test_idx \
    --rv ../res/voc \
    --pm ../model/sl.model \
    --ml 500 \
    --bs 64 \
    -g \
    --device_ids 2 \
    -o ../data/result.txt
