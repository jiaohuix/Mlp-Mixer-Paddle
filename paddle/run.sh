#/bin/bash
stage=0
CONFIG_FILE="conf/base.yaml"
# 获取stage
if [ ! -z $1 ];then
  stage=$1
fi

# 训练
if [ $stage -le 0 ];then
  echo ================================================================================
  echo "                    Stage 1: Training Cifar10 Classifier... "
  echo ================================================================================
  python main.py --config $CONFIG_FILE --mode train  || exit 1
fi

# 评估
if [ $stage -le 1 ];then
  echo ================================================================================
  echo "                    Stage 2: Evaluating Cifar10 Classifier...      "
  echo ================================================================================
    python main.py --config $CONFIG_FILE --mode eval|| exit 1
fi

