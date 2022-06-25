# install
cd /workspace/AdelaiDet
pip3 install opencv-python
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python setup.py build develop


# train
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file ./configs/BoxInst/MS_R_50_1x_leaf.yaml  \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/boxinst_leaf 

# evaluate
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file ./configs/BoxInst/MS_R_50_1x_leaf.yaml  \
    --eval-only \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/boxinst_leaf    \
    MODEL.WEIGHTS /workspace/AdelaiDet/training_dir/boxinst_leaf/model_final.pth

# inference
python demo/demo.py \
    --config-file ./configs/BoxInst/MS_R_50_1x_leaf.yaml  \
    --input ./datasets/images/test \
    --output ./viz/image \
    --opts MODEL.WEIGHTS training_dir/boxinst_leaf/model_0044999.pth

