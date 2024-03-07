Visual accumulator model (VAM)
------------

The VAM is a probabilistic model of raw (pixel-space) visual stimuli and response time (RT) data from cognitive tasks. A convolutional neural network (CNN) takes visual stimuli as inputs and outputs the drift rates for an evidence accumulation model (EAM) of speeded decision-making. The CNN and EAM parameters are jointly fitted in a variational Bayesian framework. The VAM allows one to characterize how abstract, task-relevant representations are extracted from raw perceptual inputs, and how these representations relate to behavior.

Preprint: [placeholder]\
Data: [placeholder]

Reproducing the paper
------------

1) Create and activate a new Python3.10 environment, and install the analysis dependencies. From the main repo folder:

```
python3.10 -m venv analysis_env
source analysis_env/bin/activate
pip install -r analysis_requirements.txt
```

2) Download the trained models, metadata, and derivatives from Zenodo, linked above. These should be placed in one directory with the following subfolders: vam_models, task_opt_models, binned_rt_models, and derivatives; and the following files: metadata.csv and binned_rt_metadata.csv. For convenience, consider setting the overarching directory as an environment variable $MODEL_DIR (referenced below).

3) Run the model/participant analyses using the run_model_analysis.py CLI. From the manuscript folder:

```
python run_model_analysis.py $MODEL_DIR -s "my_summary_stats"
```

This will run all of the analyses of the model representations/activations and model/participant behavior, and save summary analyses in MODEL_DIR/derivatives/my_summary_stats.

4) Create the manuscript figures using the make_manuscript.py CLI. From the manuscript folder:

```
python make_manuscript.py $MODEL_DIR -s "my_summary_stats"
```

All of the main/supplemental figures and summary stats .txt files will be saved in MODEL_DIR/derivatives/my_summary_stats/figures. Use the -f flag to create specific figures. E.g. to create figures 3 and S5:

```
python make_manuscript.py $MODEL_DIR -s "my_summary_stats" -f 3 S5
```

5) The steps above will run the summary analyses on precomputed model outputs (i.e. RTs/choices generated from the the test set image stimuli). To create the model outputs de novo, follow the steps below.

Training models and creating image stimuli
------------

### Creating visual stimuli/model inputs
To build the visual stimuli and preprocess the model inputs:

1) Create/activate a new Python3.10 virtual environment and install preprocessing_requirements.txt

2) Download the gameplay_data and graphics directories and metadata.csv file linked above. Put these in the same directory (e.g. $MODEL_DIR).

3) Run the make_model_inputs.py CLI, optionally with the -u flag set to specify particular users to process:

```
python make_model_inputs.py $MODEL_DIR -u 677 166
```

The outputs will be saved in $MODEL_DIR/model_inputs.

If you just want to try training a model, you can skip this step and use the example preprocessed model inputs in example_model_inputs available on Zenodo.

### Training models
These instructions assume you have a CUDA-compatible GPU with the appropriate drivers installed. Training on CPU should be possible, but I haven't tested it (and it will be much slower).

1) Create and activate a new virtual environment to install the training dependencies. I've only tested Python3.10, so 3.10 is strongly encouraged.

2) Install JAX version 0.4.11 for your CUDA version and OS. E.g. for CUDA 11 and linux:

```
pip install --upgrade "jax[cuda11_pip]==0.4.11" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Some newer versions of JAX conflict with the other dependencies, so I'd recommend sticking with version 0.4.11. For other OS, follow the installation instructions on the JAX website.

3) Install the other training dependencies listed in training_requirements.txt:

```
pip install -r training_requirements.txt
```

4) Now you're ready to train a VAM! The train_model.py CLI will train a VAM with the default parameters using the provided data directory, save directory, and experiment/run name:

```
python train_model.py "/data/path" "/save/path" "my_experiment"
```

The data directory should contain the image and RT data from one participant, e.g. the provided fully-preprocessed dataset mentioned above. Metadata and checkpoints for the training run will be saved in the save directory. See train_model.py for other options and training other types of models (e.g. task-optimized models).

### Generating de novo model outputs used in the paper
To generate RTs/choices from the holdout stimuli for all models, activate the training environment described above, and run:

```
python make_model_inputs.py $MODEL_DIR
```

Use the -u flag to generate RTs from a subset of users, e.g.:

```
python make_model_inputs.py $MODEL_DIR -u 677 166
```


