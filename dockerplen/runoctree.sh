export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/Plenoctree/checkpoints/syn_sh16
export SCENE=lego
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.extraction \
    --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree.npz
