# Concept Gradients

This is the official repository for the paper [Concept Gradient: Concept-based Interpretation Without Linear Assumption](https://openreview.net/forum?id=_01dDd3f78) published in ICLR 2023. Concept gradients is a drop-in replacement for Concept Activation Vectors (CAVs) to improve the representation of concept with non-linear functions while following the same line of intuition for constructing attribution. You can find the code to reproduce all the experiment results.

## Quick start

Setup environment

```
conda env create -n cg python
conda activate cg
pip install -e .
```

The scripts for training and attributing various models and datasets are in `./scripts`. 

## Running Concept Gradients on your custom models and datasets

Running concept gradients on your custom models and datasets is rather simple. [`src/cg/concept_gradients.py`](https://github.com/jybai/concept-gradients/blob/main/src/cg/concept_gradients.py) is a standalone file that performs the full functionality of attributing with concept gradients. You can download just this file to your local directory and make sure both `torch` and `captum` are installed as prerequisites. Our implementation depends on the [Captum](https://captum.ai) library to calculate input gradients and with respect to a certain layer. See the commets in file for more argument details. Example usage of concept gradients can be found [here](https://github.com/jybai/concept-gradients/blob/main/src/cg/attribute_cg.py#L76).

## Citation

Please cite our work if you find this repo useful

```
@inproceedings{
  bai2023concept,
  title={Concept Gradient: Concept-based Interpretation Without Linear Assumption},
  author={Andrew Bai and Chih-Kuan Yeh and Neil Y.C. Lin and Pradeep Kumar Ravikumar and Cho-Jui Hsieh},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=_01dDd3f78}
}
```

