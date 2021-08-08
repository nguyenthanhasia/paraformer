# Paraformer
This repository contains the code for the paper: Tentative

# Resource


*   Data provided by COLIEE Competition
*   Pretrained checkpoint can be downloaded [here](https://github.com/nguyenthanhasia/paraformer/releases/download/0.2/Paraformer.ckpt)


## Installation


```
conda create -n paraformer python=3.6
conda activate paraformer
pip install -r requirements.txt
```

## Training

```
python train.py \
  --data-dir DATA_DIR \
  --test-file TEST_FILE \
  --max-epochs NUM_EPOCHES
```

## Evaluating

```
python eval.py \
  --data-dir DATA_DIR \
  --test-file TEST_FILE \
  --checkpoint CHECKPOINT_PATH \
  --bm25-top-n TOP_N \
  --alpha ALPHA
```
