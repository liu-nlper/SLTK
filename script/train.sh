#Options:
#  -h, --help            show this help message and exit
#  -f FEATURES           使用的特征列数
#  --root_idx_train=ROOT_IDX_TRAIN
#                        训练数据索引根目录
#  --root_idx_dev=ROOT_IDX_DEV
#                        开发数据索引根目录
#  --rv=ROOT_VOC         字典根目录
#  --re=ROOT_EMBED       embed根目录
#  --ml=MAX_LEN          实例最大长度
#  --ds=DEV_SIZE         开发集占比
#  --lstm=LSTM           LSTM单元数
#  --ln=LAYER_NUMS       LSTM层数
#  --fd=FEATURE_DIM      输入特征维度
#  --dp=DROPOUT          dropout rate
#  --lr=LEARN_RATE       learning rate
#  --ne=NB_EPOCH         迭代次数
#  --mp=MAX_PATIENCE     最大耐心值
#  --rm=ROOT_MODEL       模型根目录
#  --bs=BATCH_SIZE       batch size
#  -c, --char            是否使用字符特征
#  --dc=DIM_CHAR         字符特征维度
#  --fs=FILTER_SIZES
#                        卷积核尺寸(char feature)
#  --fn=FILTER_NUMS
#                        卷积核数量(char feature)
#  --seed=SEED           随机数种子
#  -g, --cuda            是否使用GPU加速
#  --crf                 是否使用CRF层
#  --nw=NB_WORK          加载数据的线程数


CUDA_VISIBLE_DEVICES=3 python3 ../train.py \
    -f 0,1 \
    --root_idx_train ../data/train_idx \
    --root_idx_dev ../data/dev_idx \
    --rv ../res/voc \
    --ml 55 \
    --fd 200,32 \
    --re ../res/embed \
    --ds 0.1 \
    --lstm 128 \
    --lr 0.02 \
    --ne 1000 \
    --mp 5 \
    --bs 64 \
    --dp 0.65 \
    --rm ../model \
    -c \
    --c_binary \
    --dc 16 \
    --fs 3 \
    --fn 32 \
    --device_ids 3 \
    --rg False \
    --crf \
    --nw 16 \
    -g
