
CUDA_VISIBLE_DEVICES=0 python train.py --config="./kqq_configs/01a.yaml" --no_log

CUDA_VISIBLE_DEVICES=0 python inference.py \
	--config="./kqq_configs/01a.yaml" \
	--ckpt_path="./checkpoints/train/01a/step=200000.pth" \
	--audio_path="./assets/cut_liszt.mp3" \
	--midi_path="_zz.mid"

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
	--config="./kqq_configs/01a.yaml" \
	--ckpt_path="./checkpoints/train/01a/step=200000.pth" \
	--results_dir="./results/01a"


#
CUDA_VISIBLE_DEVICES=0 python train_slakh.py --config="./kqq_configs/slakh_01a.yaml" --no_log


# train.py 		piano transcriptions
# train_slakh.py  multi-track transcription

# 01a.yaml		maestro, conformer2d
# 02a.yaml		maestro, conformer2d, lr=3e-4

# slakh_01a.yaml  conformer2d_nopool
# slakh_02a.yaml  conformer2d_nopool, 2s

# 04a.yaml		maestro, transformer, works
# 05a.yaml		maestro, FlextokContinuous
# 06a.yaml		maestro, FlextokFSQ

# 07a.yaml		maestro, FlextokContinuous2, pos emb, long + local transformer
# 08a.yaml		maestro, FlextokFSQ2, pos emb, long + local transformer
# 08b.yaml		maestro, FlextokFSQ2, pos emb, long + local transformer, lr=1e-4