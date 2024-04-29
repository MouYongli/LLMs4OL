# LLMs4OL
Large Language Models for Ontology Learning

## Dataset Setup



## Python Environment Setup

1. conda environment
```
conda create --name=llms4ol python=3.10
conda activate llms4ol
```

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
pip install -r requirements.txt
pip install -e .
```

2. jupyter lab and kernel
```
conda install -c conda-forge jupyterlab
conda install ipykernel
ipython kernel install --user --name=llms4ol
```

exit and reopen a session (conda env llms4ol)

```
jupyter lab --no-browser --port=8888
```
