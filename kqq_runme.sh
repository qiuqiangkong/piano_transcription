
CUDA_VISIBLE_DEVICES=2 python train_llama_ft.py

CUDA_VISIBLE_DEVICES=2 python inference_llama.py

# Finetune onset, offset, vel, frame
CUDA_VISIBLE_DEVICES=2 python train_llama_ft2.py

# Finetune onset, offset, vel, frame, deeper LM
CUDA_VISIBLE_DEVICES=2 python train_llama_ft3.py

# Finetune onset, offset, vel, frame, deeper LM, multiple loss
CUDA_VISIBLE_DEVICES=2 python train_llama_ft4.py

# Train vel
CUDA_VISIBLE_DEVICES=0 python train_llama_vel.py

CUDA_VISIBLE_DEVICES=1 python train_llama_off.py