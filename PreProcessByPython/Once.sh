#!/bin/sh
# 训练步骤
python3 train.py ./CC/MFCC CC.gmm ./GMM-model
python3 train.py ./SC/MFCC SC.gmm ./GMM-model
# 测试步骤
python3 test.py ./GMM-model ./SC-Test/MFCC ./GMM-model/SC
python3 test.py ./GMM-model ./CC-Test/MFCC ./GMM-model/CC