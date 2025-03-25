# Repository for implementing the CRSA model

## Configuration

Create a new enviroment and install the project's requirements:

```bash
conda create --name crsa python=3.12.7
conda activate crsa
pip install -r requirements.txt
pip install -e .
```


# Scripts:
- rsa_hyperparams.py: Run RSA on single world for different alphas and show evolution metrics (gain, cooperation, etc.) and performance of the model compared to baselines.
- rsa_dataset.py: Run RSA on real dataset and compare performance with baselines.
- yrsa_hyperparams.py and yrsa_dataset.py: Repeat for Y-RSA
- collaborative_models_hyperparms.py: Run CRSA and Mutiple Y-RSA and PIP on single world for different alphas and show:
    - evolution metrics (gain, cooperation, etc.) for each round
    - final metric (gain, cooperation, etc.) vs rounds 
    - performance of the model compared to:
        - dummy (priors)
        - yrsa using last utterance
        - multi y-rsa
- collaborative_models_dataset.py: Run CSRA, Multiple Y-RSA and PIP for best parameters on real dataset.

