# LLMs4OL
Large Language Models for Ontology Learning

## Table of Contents
- [Repository Structure](#repository-structure)
- [Dataset Setup](#dataset-setup)
- [Python Environment Setup](#python-environment-setup)


## Repository Structure
```
.
├── LICENSE
├── Makefile
├── README.md         
├── requirements.txt                    <- list all dependency packages required by the project
├── setup.py           
└── src                                 <- root directory of the repository                 
    ├── assets	                        <- artifacts directory
    │   ├── Datasets                    <- contains original/processed datasets for pretraining and fintuning
    │   │   ├── ...         
    │   ├── LLMs                        <- contains pretrained LLMs 
    │   └── Tuning                      <- contains fine-tuned LLMs for each subtask
    ├── llms4ol
    │   ├── Analyse                     <- contains figures and its generating methods for analysing the results and presentations
    │   ├── DataProcess                 <- contains dataset builder for each task / data domain
    │   │   ├── ...
    │   ├── Evaluate                    <- contains functions for evaluating pretrained/finetuned LLMs
    │   │   ├── ...
    │   └── Training                    <- contains functions for pretraining and finetuning
    │       ├── ...

```


## Dataset Setup



## Python Environment Setup

### 1. Conda Environment Set Up
```
conda create --name=llms4ol python=3.10
conda activate llms4ol
```
- Use ``nvcc --version`` to check the version of CUDA. CUDA 12+ is required.
- You may need to check your GPU driver for compatibility issues with CUDA Toolkit, please refer to [this link](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) to find the information in NVIDIA's official documentation.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -U g4f[all]
pip install -e .
```

### 2. Jupyter lab and kernel (Optional)
```
conda install -c conda-forge jupyterlab
conda install ipykernel
ipython kernel install --user --name=llms4ol
```

- Exit and reopen a session

```
conda activate llms4ol
jupyter lab --no-browser --port=9000
```
- Once Jupyter Server is running remotely, open another session. Keep 2 sessions alive in background.

```
ssh -L 8889:localhost:9000 name@ip
```

