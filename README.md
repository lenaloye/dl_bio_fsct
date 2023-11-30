# Few Shot Learning - Cosine Transfomer
## Project for Deep Learning in Biomedicine course
Group 11 - Leonard Treil and Lena Loye

## Installation

### GitHub repository
Clone the repository and navigate to the dl_bio_fsct folder.

```bash
git clone https://github.com/...git
```

### Dataset
Create a data folder in the root of the project
Download the BreaKHis dataset from kaggle: https://www.kaggle.com/datasets/ambarish/breakhis
Extract the downloaded zip file to the data folder

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
Algorithm implementations based on [COMET](https://github.com/snap-stanford/comet) and [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot). Dataset preprocessing code is modified from each respective dataset paper, where applicable.

