# DockGame: Cooperative Games for Multimeric Rigid Protein Docking

### [arXiv](https://arxiv.org/)

Implementation of **DockGame**, a game-theoretic framework for multimeric (rigid) protein docking.
We model docking as a cooperative game between proteins where the assembly structures correspond
to equilibria. To compute equilibria for new proteins, we propose to learn the underlying potential
in two ways:

1. Learning a surrogate game potential guided by PyRosetta
2. Learning a diffusion model over the action spaces of all agents

The former approach learns a differentiable analogue of traditional scoring functions for docking,
while the latter approach connects equilibrium computation to sampling for cooperative games.

The current models were trained on a subset of DIPS dataset and finetuned on DB5.5.

For using **DockGame** on your own multimeric complexes: We are currently 
compiling a larger dataset using all assemblies from PDB to train our score and
potential models, and will update the repository with the latest versions once 
complete.

![dockgame-overview](./assets/dockgame.png)

## Installation

### Environment
To install the conda environment and necessary packages, run the following command

```bash
./build_env.sh
```
The installation should work on Linux, Mac and M1/M2 Mac.

### PyRosetta

PyRosetta is required to compute binding energies for generated decoys. Please follow the instructions 
[**here**](https://www.pyrosetta.org/downloads) to install, and store the outputs in `bin/`

Note: For Linux, it is easier to download the `.whl` file to the appropriate python version and just run

```
pip install <path to .whl file>
```

To install the distributed version of pyrosetta, run:

```
conda install pyrosetta distributed -c https://USERNAME:PASSWORD@conda.graylab.jhu.edu
pip install blosc
```

where `USERNAME` and `PASSWORD` are provided by PyRosetta with the license file.

## Datasets 

By default, we assume that datasets are stored under `data/raw` for the raw datasets
and `data/processed/` for the processed datasets.

### Downloading Datasets

All datasets used in this work can be found on [zenodo](https://zenodo.org/record/8408573).

To download the raw data and extract it, run the following command:
```
bash download_data.sh DIRNAME
```

By default, this stores the downloaded datasets under `data/raw/` but you could
specify an optional `DIRNAME`.

### Processing Datasets

To process the datasets, run the following command:

```
python scripts/preprocess/prepare_complexes.py \
    --dataset DATASET \
    --data_dir DATA_DIR \
    --complex_list_file COMPLEX_LIST_FILE \
    --complex_dir COMPLEX_DIR \
    --agent_type AGENT \
    --featurizer base \
    --resolution c_alpha \
    --center_complex
```

where `DATASET` is one of `db5,dips` and `AGENT` is one of `protein,chain`. In 
this work, we set `AGENT=chain`. 

`COMPLEX_DIR` refers to the directory where assembly structures are stored (under `DATA_DIR/raw/DATASET/COMPLEX_DIR`)

`COMPLEX_LIST_FILE` refers to the list of complexes to process (under `DATA_DIR/raw/DATASET`)

### Generating Decoys

For training the potential network, we generate decoys using `PyRosetta`. 
This can be done with the following command:

```
python scripts/preprocess/generate_decoys.py \
    --dataset DATASET \
    --agent_type AGENT \
    --complex_list_file COMPLEX_LIST_FILE \
    --complex_dir COMPLEX_DIR
    --score_fn_name dock_low_res
    --max_tr MAX_TR
```

where `MAX_TR` is the maximum magnitude of translation for generating decoys.

## Training

### Potential Model

For training the potential model, run the following command

```
python scripts/train/reward.py --config CONFIG_FILE
```

while `CONFIG_FILE` is the config file. Please refer to `dockgame/utils/setup.py`
and the `model_parameters.yml` file under `paper/potential_model` for associated arguments.

### Score Model

For training the score model, run the following command

```
python scripts/train/score.py --config CONFIG_FILE
```

while `CONFIG_FILE` is the config file. Please refer to `dockgame/utils/setup.py`
and the `model_parameters.yml` file under `paper/score_model` for associated arguments.


## GamePlay

To run inference or game-play, run the following commands depending on the model

```
python scripts/gameplay.py \
    --dataset DATASET \
    --data_dir DATA_DIR \
    --complex_dir COMPLEX_DIR \
    --complex_list_file COMPLEX_LIST_FILE \
    --task_name test-gameplay \
    --featurizer base \
    --model_dir paper/score_model \
    --model_name model.pt \
    --n_rounds N_ROUNDS \
    --n_equilibria N_EQUILIBRIA \
    --strategy STRATEGY \
    --agent_type AGENT
```

`STRATEGY` is one of `langevin, round_robin_langevin` for the score model 
and `reward_grad, round_robin_reward_grad` for the potential model. 
We also include the option to save visualization by using `--save_visualization`.
By default, this saves the structures after initialization, and after inference.

This loads the structures from `COMPLEX_DIR`, computes features and prepares graph,
and runs gameplay, and optionally saves the final structures.

We also include a script with more logging, suited for debugging (`scripts/gameplay_debug.py`).
This script also includes an option to save full trajectories. For this enable
`--save_vis, --debug, --save_trajectory`.

## Reproducibility

We provide the models used in this paper under `paper/`. To evaluate either model, 
run the following commands (this is for the score model):

```
python scripts/gameplay.py \
    --dataset db5 \
    --out_dir game_outputs \
    --data_dir data \
    --complex_dir complexes \
    --complex_list_file complexes-test.txt \
    --task_name db5-score-evaluation \
    --featurizer base \
    --model_dir paper/score_model \
    --model_name model.pt \
    --n_rounds 50 \
    --n_equilibria 40 \
    --strategy langevin \
    --agent_type chain

python scripts/evaluate.py \
    --dataset db5 \
    --results_dir game_outputs \
    --task_name db5-score-evaluation \
    --complex_list_file complexes-test.txt \
    --complex_dir complexes \
    --data_dir data
```

## License

The project is licensed under the MIT License.

## Reference

If you find our work useful, please cite our paper:

## Contact

If you have any questions about the code, or want to report a bug, 
or need help interpreting an error message, please raise a GitHub issue.

## Contribution

This code was contributed by Vignesh Ram Somnath and Pier Giuseppe Sessa.