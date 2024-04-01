
# A Minimal Implementation of Piano Transcription

0. Install dependencies

```bash
git clone https://github.com/qiuqiangkong/mini_piano_transcription

# Install Python environment
conda create --name piano_transcription python=3.8

conda activate piano_transcription

# Install Python packages dependencies.
sh env.sh

```

# Train
CUDA_VISIBLE_DEVICES=0 python train.py

# Inference
CUDA_VISIBLE_DEVICES=0 python inference.py

# Evaluate
python evaluate.py