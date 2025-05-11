# simcse-ablations

## Project Overview

The aims of our project are to validate the author’s claims by reproducing the BERT-base SimCSE ([Gao et al., 2021](https://arxiv.org/abs/2104.08821)) model's results on the same semantic textual similarity (STS) tasks, as well as to propose and experiment with four new model ablations to improve different components of the model:

**Batch-Aware Linear Dropout Schedule**  
Adjusts the dropout rate linearly based on the training step within each batch, allowing finer control over regularization dynamics.

**Layer-Aware Linear Dropout Schedule**  
Assigns dropout rates that vary across encoder layers to regularize layers differently, with the goal of preserving information in lower layers while regularizing higher layers more aggressively.

**Different Pooling Methods**  
Investigates alternatives to `[CLS]` pooling, including mean pooling, max pooling, first/last-layer averaging, and attention-based pooling, to assess their impact on embedding quality.

**Temperature Control**  
Explores learnable or adaptive temperature scaling in the contrastive loss to better calibrate embedding similarity distributions.



## Hyperparameters

Below are the key hyperparameters used for training and ablation experiments. The hyperparmeters are set in the first cell of the notebook.

```
# General training settings
batch_size    = 64         # Number of samples per batch
seed          = 49         # Random seed for reproducibility
num_epochs    = 1          # Total number of training epochs
learning_rate = 3e-5       # Optimizer learning rate
temperature   = 0.05       # Temperature for contrastive loss
max_length    = 32         # Maximum tokenized sequence length
num_samples   = 1000000    # Number of training samples (set to 1M for full training)
log_every     = 100        # Logging interval (in steps)

# Batch-aware dropout schedule
p_start = 0.1              # Dropout rate at start of batch
p_end   = 0.1              # Dropout rate at end of batch

# Layer-aware dropout schedule
dropout_start = 0.1        # Dropout for lower layers
dropout_end   = 0.1        # Dropout for higher layers

# Pooling configuration
pooling_method = 'self_attention'  # Pooling method for sentence embeddings

# Temperature scaling options
learn_temp       = False    # Whether temperature is learnable
use_dynamic_temp = False    # Whether temperature changes during training

```

### Semantic Textual Similarity (STS) Results

The following table compares our model's performance (using the default hyperparameters set above) with the original [SimCSE](https://arxiv.org/abs/2104.08821) results on seven STS tasks.

|          Method | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg.   |
|----------------|-------|-------|-------|-------|-------|--------|--------|--------|
| **Ours**       | 68.17 | 82.79 | 74.53 | 81.97 | 77.97 | 77.95  | 70.76  | **76.16** |
| **Theirs**     | 68.40 | 82.41 | 74.38 | 80.91 | 78.56 | 76.85  | 72.23  | **76.25** |


