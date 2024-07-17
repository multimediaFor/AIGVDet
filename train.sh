EXP_NAME="moonvalley_vos2_crop"
DATASETS="moonvalley_vos2_crop"
DATASETS_TEST="moonvalley_vos2_crop"
python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST 