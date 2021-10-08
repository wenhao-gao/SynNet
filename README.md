# synth_net

This repo contains the code and analysis scripts for generating synthetic trees. Our model can serve as synthesis planning tools and synthesizable molecular designer.

## Setup instructions

### Setting up the environment
You can use conda to create an environment containing the necessary packages and dependencies for running synth_net by using the provided YAML file:

```
conda env create -f env/synthenv.yml
```

If you update the environment and would like to save the updated environment as a new YAML file using conda, use:

```
conda env export > path/to/env.yml
```

### Installation
[TODO this is not currently the way to install the package]
To install the package, type from the main `synth_net/` dir (where `setup.py` is located):
```
pip install -e .
```

### Unit tests
To check that everything has been set-up correctly, you can run the unit tests from within the `tests/` directory by typing:

```
python -m unittest
```

You should get no errors if everything ran correctly.

## Code Structure
The code is structured as follows:

```
synth_net/
├── data
│   ├── rxn_set_hb.txt
│   └── rxn_set_pis.txt
├── environment.yml
├── LICENSE
├── README.md
├── scripts
│   ├── compute_embedding_mp.py
│   ├── compute_embedding.py
│   ├── generation_fp.py
│   ├── generation.py
│   ├── gin_supervised_contextpred_pre_trained.pth
│   ├── _mp_decode.py
│   ├── _mp_predict_beam.py
│   ├── _mp_predict_multireactant.py
│   ├── _mp_predict.py
│   ├── _mp_search_similar.py
│   ├── _mp_sum.py
│   ├── mrr.py
│   ├── optimize_ga.py
│   ├── predict-beam-fullTree.py
│   ├── predict_beam_mp.py
│   ├── predict-beam-reactantOnly.py
│   ├── predict_mp.py
│   ├── predict_multireactant_mp.py
│   ├── predict.py
│   ├── read_st_data.py
│   ├── sample_from_original.py
│   ├── search_similar.py
│   ├── sketch-synthetic-trees.py
│   ├── st2steps.py
│   ├── st_split.py
│   └── temp.py
├── setup.py
├── synth_net
│   ├── data_generation
│   │   ├── check_all_template.py
│   │   ├── filter_unmatch.py
│   │   ├── __init__.py
│   │   ├── make_dataset_mp.py
│   │   ├── make_dataset.py
│   │   ├── _mp_make.py
│   │   ├── _mp_process.py
│   │   └── process_rxn_mp.py
│   ├── __init__.py
│   ├── models
│   │   ├── act.py
│   │   ├── mlp.py
│   │   ├── prepare_data.py
│   │   ├── rt1.py
│   │   ├── rt2.py
│   │   └── rxn.py
│   └── utils
│       ├── data_utils.py
│       ├── ga_utils.py
│       └── __init__.py
└── tests
    ├── create-unittest-data.py
    └── test_DataPreparation.py
```

The model implementations can be found in `synth_net/models/`, with processing and analysis scripts located in `scripts/`. 

## Running the code
[TODO for each of the subsections below, will be good to add time/RAM estimates]
Before running anything, you need to add the root directory into python path. One option is to run in the root synth_net directory:

```
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Processing the data: reaction templates and applicable reactants
Given a set of reaction templates and a list of buyable building blocks, we first need to assign applicable reactants for each template. Under `synth_net/synth_net/data_generation`, run:

```
python process_rxn_mp.py
```

This will save the reaction templates and their corresponding building blocks in a JSON file. Then, run:

```
python filter_unmatch.py 
```

This will filter out buyable building blocks which didn't match a single template.

### Generating the synthetic path data by random selection
Under `synth_net/synth_net/data_generation`, run:

```
python make_dataset_mp.py
```

This will generate synthetic path data saved in a JSON file. Then, to make the dataset more pharmaceutically revelant, we can change to `synth_net/scripts/` and run:

```
python sample_from_original.py 
```

This will filter out the samples where the root node QED is less than 0.5, or randomly with a probability less than 1 - QED/0.5.

### Splitting data into training, validation, and testing sets, and removing duplicates
Under `synth_net/scripts/`, run:

```
python st_split.py
```

The default split ratio is 6:2:2 for training, validation, and testing sets.

### Featurizing data
Under `synth_net/scripts/`, run:

```
python st2steps.py -r 2 -b 4096 -d train
```

This will featurize the synthetic tree data into step-by-step data which can be used for training. The flag *-r* indicates the fingerprint radius, *-b* indicates the number of bits to use for the fingerprints, and *-d* indicates which dataset split to featurize. 

### Preparing training data for each network
Under `synth_net/synth_net/models/`, run:

```
python prepare_data.py --radius 2 --nbits 4096
```

This will prepare the training data for the networks.

Each is a training script and can be used as follows (using the action network as an example):

```
python act.py --radius 2 --nbits 4096
```

This will train the network and save the model parameters at the state with the best validation loss in a logging directory, e.g., **`act_hb_fp_2_4096_logs`**. One can use tensorboard to monitor the training and validation loss.

### Reconstructing a list of molecules
To test how good the trained model is at reconstructing from a set of known molecules, we can evaluate the model for the task of single-shot retrosynthesis.

[TODO add checkpoints to prediction scripts // save trees periodically. otherwise just saves at end and is problematic of job times out]
```
python predict.py --radius 2 --nbits 4096
``` 

This script will feed a list of molecules from the test data and save the decoded results (predicted synthesis trees) to `synth_net/results/`.

Note: this file reads parameters from a directory with a name such as **`hb_fp_vx`**, where "hb" indicates the Hartenfeller-Button dataset, "fp" indicates to use the fingerprint featurization (as opposed to GIN embeddings), and "vx" indicates the version (x in this case).
### Molecular optimization
Under `synth_net/scripts/`, run:

```
python optimization_ga.py
```

This script uses a genetic algorithm to optimize molecular embeddings and returns the predicted synthetic trees for the optimized molecular embedding.

### Sketching synthetic trees
To visualize the synthetic trees, run:

```
python scripts/sketch-synthetic-trees.py --file /pool001/whgao/data/synth_net/st_hb/st_train.json.gz --saveto ./ --nsketches 5 --actions 3
```

This will sketch 5 synthetic trees with 3 or more actions to the `./synth_net/` directory (you can play around with these variables or just also leave them out to use the defaults).

### Testing the mean reciprocal rank (MRR) of reactant 1
Under `synth_net/scripts/`, run

```
python mrr.py --distance cosine
```
