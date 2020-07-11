
# CATN

Codes for SIGIR 2020 paper [CATN: Cross-Domain Recommendation for Cold-Start Users via
Aspect Transfer Network](https://arxiv.org/abs/2005.10549).

## Citation

Please cite our paper if you find this code useful for your research:

```
@inproceedings{sigir20:catn,
  author    = {Cheng Zhao and
               Chenliang Li and
               Rong Xiao and
               Hongbo Deng and
               Aixin Sun},
  title     = {{CATN:} Cross-Domain Recommendation for Cold-Start Users via Aspect
               Transfer Network},
  booktitle = {{SIGIR}},
  year      = {2020},
}
```

## Requirement
* python 3.6
* tensorflow 1.10.0
* numpy
* pandas
* scipy
* gensim
* sklearn
* tqdm


## Files in the folder
- `dataset/`
  - `preprocessing.py`: constructing cross-domain datasets;
- `runner/`
  - `CATN_runner.py`: the main runner (including the configurations);
- `utils`
  - `CATN.py`: CATN implementation.


## Running the code
1. Download the original data from [Amazon-5core](http://jmcauley.ucsd.edu/data/amazon/index.html), 
choose two relevant categories (*e.g.*, Books, Movies and TV) and put them under the same directory in dataset/.

2. run python preprocessing.py.

3. run python CATN_runner.py.