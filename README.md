# Piano Transcription with Neural Networks

This repository is a PyTorch implementation of piano transcription systems. The system takes audio as input and outputs onset, offset, frame, and velocity rolls. Users can train the system in less than 10 hours using a single RTX 4090 GPU.

## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/piano_transcription
cd piano_transcription

# Install Python environment
conda create --name piano_transcription python=3.10

# Activate environment
conda activate piano_transcription

# Install Python packages dependencies
bash env.sh
```

## 1. Download dataset

Users need to do download the Maestro dataset (131 GB, 199 hours).

```bash
bash ./scripts/download_maestro.sh
```

The downloaded dataset after compression looks like:

<pre>
maestro-v3.0.0 (131 GB)
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

## 2. Train

Takes 10 hours on 1 RTX4090 to train for 100,000 steps

```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/conformer2d.yaml"
```

### 3. Inference

If user has his own piano recording. Downloaded a pretrained checkpoint for inference. Or use a checkpoint trained by yourself for inference.

```bash
mkdir -p ./checkpoints/train/conformer2d
wget -O ./checkpoints/train/conformer2d/step=200000.pth https://huggingface.co/qiuqiangkong/piano_transcription/resolve/main/conformer2d_step%3D200000.pth?download=true
```

```python
CUDA_VISIBLE_DEVICES=0 python inference.py \
	--config="./configs/conformer2d.yaml" \
	--ckpt_path="./checkpoints/train/conformer2d/step=200000.pth" \
	--audio_path="./assets/cut_liszt.mp3" \
	--midi_path="./results/cut_liszt.midi"
```

The transcribed result can be listened at [./results/cut_liszt.midi](https://github.com/qiuqiangkong/piano_transcription/results/cut_liszt.midi)

## 4. Evaluate

Evaluate the precision, recall, and F1 scores on the test set of the MAESTRO dataset.

```python
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
	--config="./configs/conformer2d.yaml" \
	--ckpt_path="./checkpoints/train/conformer2d/step=200000.pth" \
	--results_dir="./results/conformer2d"
```

<pre>
0/177, /datasets/maestro-v3.0.0/2009/MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_07_WAV.wav                                                                                   
Write out to results/conformer2d/MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_07_WAV.midi                                                                                              
Precision: 0.9201, Recall: 0.9098, F1: 0.9149                                                                                                                                                             
35/177, /datasets/maestro-v3.0.0/2015/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_13_R2_2015_wav--4.wav                                                                                           
Write out to results/conformer2d/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_13_R2_2015_wav--4.midi                                                                                                       
Precision: 0.9182, Recall: 0.8825, F1: 0.9000                                                                                                                                                             
...
------ Average metric ------                                                                                                                                                                              
Precision: 0.9433, Recall: 0.9345, F1: 0.9388
</pre>

## License

MIT

## Cite
<pre>
@article{kong2021high,
  title={High-resolution piano transcription with pedals by regressing onset and offset times},
  author={Kong, Qiuqiang and Li, Bochen and Song, Xuchen and Wan, Yuan and Wang, Yuxuan},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={29},
  pages={3707--3717},
  year={2021},
  publisher={IEEE}
}
</pre>