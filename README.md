#  A Decomposable Attention Model for Natural Language Inference

This repository contains a `Keras` implementation of Parikh's *et al.* (2016) algorithm for natural language inference. In theory, it should be relatively straightforward to code, if it were not for some limitations with respect to masking in the present version of `Keras`, and the selection of the proper optimization algorithm along with fine-tuning the hyperparameters.

## Requirements

- Python 3.6
- Keras 2+
- Spacy 1.9 (not 2+)
- Pandas + NumPy

The code relies on the older implementation of Spacy. However, it should be relatively easy to port to the newer version in due time. The Tensorflow backend was used.

## Zero-padding and masking

Careful design is required when dealing with inputs which for practical reasons have to potentially be padded with zeros up to a specified sentence length. Indeed, many of the layers in `Keras` (`keras.layers`) do not support the usage of `Masking`, in particular `Conv1D` and `GlobalAveragePooling1D`. `MaskedConv1D` and `MaskedGlobalAveragePooling1D` in `model/layers.py` were written to overcome those shortcomings. It is important to note that at the moment they do not propagate the mask to avoid any clash with following layers. 

Furthermore, the row-wise softmax transforms following the soft-alignment, here performed by the new layer `Softmax2D`, need to take into account padding with the appropriate 2D masking using the new layer `Masking2D`.

## Optimization of model's parameters

As in the paper by Parikh *et al.*, we train the model (382K parameters) on the Stanford Natural Language Inference (SNLI) dataset over 60 epochs using a cross-entropy loss with the `Adagrad` optimizer (0.01 of learning rate), batches of size 128, and fixed sentence length $l=42$. 

It could be tempting to use the popular `Adam` optimizer or the basic `SGD`, but we found `Adagrad` to be the most stable option while systematically converging to a better minimum. 

## Results

The best accuracy on the `dev` dataset achieved by the model is 85%, only 1% shy of the reported 86% by Parikh *et al.* . Using their exact hyperparameters does not converge to a better result.