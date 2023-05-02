# Paraformer
This repository contains the code for the paper: Attentive deep neural networks for legal document retrieval

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

## Citation

- Nguyen, H., Phi, M., Ngo, X., Tran, V., Nguyen, L., & Tu, M. (2022). Attentive deep neural networks for legal document retrieval. Artificial Intelligence and Law, 1-30. Springer.

BibTeX:
```bibtex
@article{nguyen2022attentive,
  title={Attentive deep neural networks for legal document retrieval},
  author={Nguyen, Ha-Thanh and Phi, Manh-Kien and Ngo, Xuan-Bach and Tran, Vu and Nguyen, Le-Minh and Tu, Minh-Phuong},
  journal={Artificial Intelligence and Law},
  pages={1--30},
  year={2022},
  publisher={Springer}
}
```
