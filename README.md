# Multi-label open-set audio classification

# Overview

An open-set dataset with varying degrees of openness and unknown class assignments is synthesized using sources from FSD-50k. 7600 sources with a present and predominant sound event from 89 classes are used, each between 0.5s and 4s in duration.

Two datasets are synthesized, one with `low` and the other with `high` openness. The openness quantifies the proportion of known to unknown classes and indicates how challenging the open-set classification task is.
The 89 classes are split into five groups are assigned to known, known unknown, or unknown unknown subsets, leading to five dataset variants.
Within each dataset variant training, validation, and test splits are synthesized with no source overlap between them.

All dataset variants are generated using Scaper, a tool to programmatically mix sounds to create reproducible soundscapes.

# Datasets

1.  Open Set Soundscapes (OSS)

A dataset of 10s clips with one to four overlapping sources placed randomly. The sources are augmented using pitch shift and time stretch. Please refer to Section 2 in the paper for more details.

2. Open Set Tagging (OST)

Using the above 10s soundscapes, 1s clips are generated by windowing the center of each event. Any events overlapped with the window are added to the clip labels.

# How to reproduce the dataset on your machine

0. Clone this repository.

1. Create a virtual env or a conda environment on your machine and install required packages using 
```pip install -r requirements.txt```
from the root directory of the repository.

2. Synthesize .jams files from OSS
This repo uses Scaper to generate the soundscapes. Since Scaper sequentially updates its internal state to generate random soundscapes, you must sequentially generate the dataset variants of OSS in order to reproduce the dataset in the [paper](https://dcase.community/documents/workshop2023/proceedings/DCASE2023Workshop_Sridhar_11.pdf) -- i.e. variant 1, 2, ..., 5. In case of space of computational or storage constraints, Variant 1 is preferred for evaluation. 

To synthesize OSS, download the source dataset from [Zenodo](10.5281/zenodo.7241704). Then synthesize OSS using the command
```python dataset/generate_oss.py --fgpath /path/to/foreground source files \ --bgpath /path/to/foreground source files --outpath /path/to/save output jams files```
from the root folder of the repository.

By default, this will generate only JAMS annotations files (no audio). These JAMS files contain all information needed to reproduce a soundscape. 

3. Synthesize OST from OSS .jams files
To synthesize 1s OST clips from OSS, use the following command
```python dataset/generate_ost.py -o {high,low} -v variant{1,2,..,5} -s {train,val,test} -p /path/to/oss``` 
where `path/to/oss` points to the base OSS directory containing openness and dataset variants. 

Since the process of generating OST is fully deterministic given OSS, you can generate any subset of any variant in any order.

# Coming soon

- Instructions to generate ground truth estimates of OST, used to train oracle models.
- Instructions and code to generate a smaller dataset OST-tiny, to address some of the limitations of OST.

# Reference

- This dataset is inspired by FSD-MIX-CLIPS

# Acknowledgement

Thanks to Simran Kaur for the initial version of tag.py.