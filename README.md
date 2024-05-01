# LLMs4OL
Large Language Models for Ontology Learning

## Dataset Setup



## Python Environment Setup

1. conda environment
```
conda create --name=llms4ol python=3.10
conda activate llms4ol
```
- Use ``nvcc --version`` to check the version of CUDA. CUDA 12+ is required.
- You may need to check your GPU driver for compatibility issues with CUDA Toolkit, please refer to [this link](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) to find the information in NVIDIA's official documentation.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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