# TSPNet

**TSPNet: Hierarchical Feature Learning via TemporalSemantic Pyramid for Sign Language Translation**

By Dongxu Li*, Chenchen Xu*,  Xin Yu, Kaihao Zhang, Benjamin Swift, Hanna Suominen, Hongdong Li

The repository contains the implementation of TSPNet. Preprocessed dataset, video features and the inference results are available at [Google Drive](https://drive.google.com/drive/folders/1oYV_k1wqGbPUhBrkLRMQb1iWKQp5P3pp?usp=sharing).

The repository is based on [fairseq](https://github.com/pytorch/fairseq) and we gratefully thank their effort.

### Rquirements

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need NVIDIA GPU and (optional) [NCCL](https://github.com/NVIDIA/nccl)
* (optional) [BPEMB](https://nlp.h-its.org/bpemb/), for preparing your own dataset 

### Install from Source

Install the project from source and develop locally:

```bash
cd fairseq
pip install --editable .
```

### Getting Started

#### Preprocessing

You may download our preprocessed dataset from [Dataset](https://drive.google.com/drive/folders/1oYV_k1wqGbPUhBrkLRMQb1iWKQp5P3pp?usp=sharing), and put them as:

```
tspnet/
├── i3d-features/
│   ├── span=8_stride=2
│   ├── span=12_stride=2
│   └── span=16_stride=2
├── data-bin/
│   └── phoenix2014T/
│       └── sp25000/
│   
├── README.md
├── run-scripts/
└── test-scripts/
```

* **i3d-features**: the i3d output features of input videos 
* **data-bin**: the preprocessed translation texts

#### (optional) Prepare your own dataset

Instead, you also may prepare your own dataset following the steps below for both input videos and the translation texts. 

1. **Text** Preprocess the translation texts using `preprocess_sign.py` to BPE, repeatedly for each split, for example:

```bash
python preprocess_sign.py --save-vecs data/processed/emb data/ori/phoenix2014T.train.de data/processed/train.de

python preprocess_sign.py data/ori/phoenix2014T.test.de data/processed/test.de
```

The script uses the general purpose German word embeddings available at [BPEMB](https://nlp.h-its.org/bpemb/) and should be installed first `pip install bpemb`.

2. **vocabulary** Run the `fairseq-preprocess` generate the dictionary file. It is optional to drop `--dataset-impl raw` to generate binarized dataset. Without invoking binarization, the only thing we need from this step is the vocabulary (dictionary) file `dict.de.txt`.

```bash
fairseq-preprocess --source-lang de --target-lang de --trainpref data/processed/train --testpref data/processed/test --destdir data-bin/ --dataset-impl raw
```

3. **Video** Prepare sign videos and the corresponding video features (e.g. by pretrained i3d networks), and create a json file for each split (e.g. train.sign-de.sign). The json file should be of the format below. It should have the same number of entries as the text file, where each entry corresponds to the sentence at the same line no in the prepared text file.

```json
[
    {
        "ident": "VIDEO_ID", 
        "size": 64  // length of video features
    }
    ...
]

```


4. Now consolidating the text files, video json files, the word embedding and vocabulary files into a folder, following the similar structure as below:

```
data-bin/
├── train.sign-de.sign
├── train.sign-de.de
│ 
├── test.sign-de.sign
├── test.sign-de.de
│ 
├── emb
└── dict.de.txt
```


#### Training

Step into the `run_scripts` folder and start training the model by:

```bash
SAVE_DIR=CHECKPOINT_PATH bash run_phoenix_pos_embed_sp_test_3lvl.sh
```

This script run the model reported best performance as in the paper. It feeds the feature pyramid with all 3 scales of features (i.e., with windowing widths of 8, 12, 16). 

#### Testing

To validate the model on the testing test, run the testing script with the checkpoints saved from the training step.
Be sure to pass the path to the checkpoint file not the folder to the parameter `CHECKPOINT`.

```bash
CHECKPOINT=CHECKPOINT_FILE_PATH bash test_phoenix_pos_embed_sp_test_3lvl.sh
```

The scripts report multiple performance results where the last line will show the ROUGE-L and BLEU-{n} as in the paper.

### Acknowledgement

Please consider citing our paper as:

``` bibtex
@inproceedings{li2020tspnet,
	title        = {TSPNet: Hierarchical Feature Learning via Temporal Semantic Pyramid for Sign Language Translation},
	author       = {Li, Dongxu and Xu, Chenchen and Yu, Xin and Zhang, Kaihao and Swift, Benjamin and Suominen, Hanna and Li, Hongdong},
	year         = 2020,
	booktitle    = {Advances in Neural Information Processing Systems},
	volume       = 33
}
```

Please also consider citing the WLASL dataset if you use the pre-trained i3d models on this general purpose dataset.

``` bibtex
@inproceedings{li2020word,
    title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
    author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision},
    pages={1459--1469},
    year={2020}
}
```


