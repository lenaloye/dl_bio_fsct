# Few Shot Learning - Cosine Transfomer
## Project for Deep Learning in Biomedicine course
Group 11 - Leonard Treil and Lena Loye

## Introduction

This project addresses a few-shot image classification problem, specifically focusing on predicting the correct histological type between eight classes of both benign and malignant breast tumors. Our objective is to obtain accurate classification with a small number of labelled examples, which will further improve diagnostic capabilities for breast cancer detection. 

The algorithm we decided to implement is a Cosine Transformer, based on the paper "Enhancing Few-shot Image Classification with Cosine Transformer" by Nguyen et al. The Cosine Transformer has a similar architecture to a standard transformer encoder block with two skip connections to preserve information, a two-layer feed-forward network, and layer normalization to reduce noise. 


## Installation

### GitHub repository
Clone the repository and navigate to the dl_bio_fsct folder.

```bash
git clone https://github.com/lenaloye/dl_bio_fsct.git
```

### Dataset
In this project we use the BreaKHis dataset. The BreaKHis dataset presents 9,109 microscopic images of breast tumor tissue. More precisely, it contains images of four histologically distinct types of benign breast tumors: adenosis (A), fibroadenoma (F), phyllodes tumor (PT), and tubular adenoma (TA), and four malignant tumors, commonly referred to as breast cancer: carcinoma (DC), lobular carcinoma (LC), mucinous carcinoma (MC), and papillary carcinoma (PC). 

#### How to download it
- Create a data folder in the root of the project
- Download the BreaKHis dataset from kaggle: https://www.kaggle.com/datasets/ambarish/breakhis
- Extract the downloaded zip file to the data folder

### Conda
Create a conda environment and install requirements.

```bash
conda env create -f environment.yml 
```

Before each run, activate the environment with:

```bash
conda activate fewshotbench
```


## Usage

### Training

```bash
python run.py exp.name={exp_name} method=fsct dataset=breakhis
```
The experiment name must always be specified.

If you want to train several models with a single command, you can use the provided `run_breakhis.sh` file. First uncomment the models that you want to test, and then run the following:
```bash
bash run_breakhis.sh
```

### Testing

The training process will automatically evaluate at the end. To only evaluate without running training, use the following:

```bash
python run.py exp.name={exp_name} method=fsct dataset=breakhis mode=test
```

## Experiment Tracking

We use [Weights and Biases](https://wandb.ai/) (WandB) for tracking experiments and results during training. 
All hydra configurations, as well as training loss, validation accuracy, and post-train eval results are logged.
To disable WandB, use `wandb.mode=disabled`. 

You must update the `project` and `entity` fields in `conf/main.yaml` to your own project and entity after creating one on WandB.

To log in to WandB, run `wandb login` and enter the API key provided on the website for your account.

## References
Algorithm implementations based on [COMET](https://github.com/snap-stanford/comet) and [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot). 

The Cosine Transformer was implemented based on the paper: Quang-Huy Nguyen, Cuong Q. Nguyen, Dung D. Le, and Hieu H. Pham. "Enhancing few-shot image
classification with cosine transformer". IEEE Access, 11:79659–79672, 2023.

Dataset preprocessing code is modified from each respective dataset paper, where applicable.

