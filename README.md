# Repository for implementing the CRSA model

## Configuration

Create a new enviroment and install the project's requirements:

```bash
conda create --name crsa python=3.12.7
conda activate crsa
pip install -r requirements.txt
pip install -e .
```

## Scripts:

To run any of the following scripts run the following command: `python -m crsa.scripts.<script_name> --arg1 arg1 --arg2 arg2...`

- `naive_reference_game`: Produce the output for the "find A1" reference game along with the plots of the paper.
- `run_mddial`: Run Llama3.2-1B-Instruct on the MDDial, produce the literal speaker values for this dataset and outputs the table of the paper. Be sure of having the correct hardware on your machine to run the model. Hardware used in our case was NVIDIA Tesla V100 with 32 GiB of RAM.

