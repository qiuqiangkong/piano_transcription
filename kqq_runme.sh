
CUDA_VISIBLE_DEVICES=2 python train_llama_ft.py

CUDA_VISIBLE_DEVICES=2 python inference_llama.py

# Finetune onset, offset, vel, frame
CUDA_VISIBLE_DEVICES=2 python train_llama_ft2.py