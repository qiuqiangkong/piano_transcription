
# A Minimal Implementation of Music tagging

This is an minimal implementation of piano music transcription with PyTorch. We use the MAESTRO V3.0.0 dataset containing 200 hours of 1,276 audio clips for training and validation. We train a convolutional recurrent neural network as classifier.

## 0. Download datasets

Users need to download the dataset from https://zenodo.org/records/3338373. After download and unzip, the dataset looks like:

<pre>
dataset_root (131 GB)
├── 2004 (132 songs, wav + flac + midi + tsv)
├── 2006 (115 songs, wav + flac + midi + tsv)
├── 2008 (147 songs, wav + flac + midi + tsv)
├── 2009 (125 songs, wav + flac + midi + tsv)
├── 2011 (163 songs, wav + flac + midi + tsv)
├── 2013 (127 songs, wav + flac + midi + tsv)
├── 2014 (105 songs, wav + flac + midi + tsv)
├── 2015 (129 songs, wav + flac + midi + tsv)
├── 2017 (140 songs, wav + flac + midi + tsv)
├── 2018 (93 songs, wav + flac + midi + tsv)
├── LICENSE
├── maestro-v3.0.0.csv
├── maestro-v3.0.0.json
└── README
</pre>

# 0. Install dependencies

```bash
git clone https://github.com/qiuqiangkong/mini_music_transcription

# Install Python environment.
conda create --name music_tagging python=3.8

# Activate environment.
conda activate mini_music_transcription

# Install Python packages dependencies.
sh env.sh
```

# 1. Single GPU training.

We use the Wandb toolkit for logging. You may set wandb_log to False or use other loggers.

```python
CUDA_VISIBLE_DEVICES=0 python train.py
```

# Multiple GPUs training.

We use Huggingface accelerate toolkit for multiple GPUs training. Here is an example of using 4 GPUs for training.

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py
```


# Reference

```
@article{kong2021high,
  title={High-resolution piano transcription with pedals by regressing onset and offset times},
  author={Kong, Qiuqiang and Li, Bochen and Song, Xuchen and Wan, Yuan and Wang, Yuxuan},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={29},
  pages={3707--3717},
  year={2021},
}
```