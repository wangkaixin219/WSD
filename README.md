# Reinforcement Learning based Weighted Sampling for Accurate Subgraph Counting on Fully Dynamic Graph Streams

This repository stores the source codes of *Weighted Sampling with Deletions* (WSD) for subgraph counting problem.

## Usage of the algorithms

### Environment

OS: Ubuntu 18.04.5

g++ version: 7.5.0

### Train WSD-L

We provide an example for training WSD-L based on `dataset/sample.edges` as follows. 

```bash
cd train
python3 main.py
```

If you want to train your own model, you can put the edge stream data in the folder `dataset/`, and update `main.py` correspondingly. 

### Compile the codes

```bash
cd algorithm/
make clean
make
```

### Experiments on synthetic datasets

In synthetic datasets, we generate graphs by [Forest Fire](https://arxiv.org/pdf/physics/0603229.pdf) model `G(n,p)`. We need to specify the number of the vertices `n` and a density parameter `p`. Besides, we need to clarify the deletion scenario (massive or light). Here is two examples. 

```bash
./wsd syn 2000000 0.5 massive
./wsd syn 2000000 0.5 light
```

### Experiments on real datasets

The graphs are already available online via [Network Repository](https://networkrepository.com/). We first download the datasets via the following commands, 

```bash
cd dataset
wget https://nrvis.com/download/data/cit/cit-patent.zip
wget https://nrvis.com/download/data/misc/com-youtube.zip
wget https://nrvis.com/download/data/soc/soc-livejournal.zip
wget https://nrvis.com/download/data/misc/web-Google.zip
```

Unzip these files. Then, files which end with `.edges` are the edge streams (insertion-only). To run the algorithm, we first enter the directory `algorithm/` and run the following commands. 

```bash
cd algorithm
./wsd real ../dataset/sample.edges massive
./wsd real ../dataset/sample.edges light
```

### Change parameters

If you are interested in explore some other properties of the algorithm (e.g., the impact of probabilities involved in deletion generations), you can find the definitions of them in `algorithm/def.h`. 

