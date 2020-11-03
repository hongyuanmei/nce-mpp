# Noise-Contrastive Estimation for Multivariate Point Processes
Source code for [Noise-Contrastive Estimation for Multivariate Point Processes (NeurIPS 2020)](https://arxiv.org/abs/2011.00717).

## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

```
@inproceedings{mei-2020-nce,
  author =      {Hongyuan Mei and Tom Wan and Jason Eisner},
  title =       {Noise-Contrastive Estimation for Multivariate Point Processes},
  booktitle =   {Advances in Neural Information Processing Systems},
  year =        {2020}
}
```

## Instructions
Here are the instructions to use the code base.

### Dependencies and Installation
This code is written in Python 3, and I recommend you to install:
* [Anaconda](https://www.continuum.io/) that provides almost all the Python-related dependencies;

Run the command line below to install the package (add `-e` option if you need an editable installation):
```
pip install .
```
It will automatically install the following important dependencies: 
* [PyTorch 1.1.0](https://pytorch.org/) that handles auto-differentiation.

### Prepare Data
The datasets used in our experiments can be downloaded from this [Google Drive directory](https://drive.google.com/drive/folders/1aBq4TCkOMLFgD7sNwRybNNs4OEIlGD8H?usp=sharing). 

Place the datasets into: 
```
data
```

### Train Models
Go to
```
ncempp/run
```

To train models with MLE or NCE, try the command line below for detailed guide:
```
python train.py --help
```

The training logs and model parameters are stored in this directory: 
```
logs/DATA_NAME/PARAMS
```

### Draw Curves
To organize training logs and draw learning curves, use the command lines below for detailed guide: 
```
python org_log.py --help
```
```
python draw_lc.py --help
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
