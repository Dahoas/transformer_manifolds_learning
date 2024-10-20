# Transformer Scaling Laws Through Intrinsic Data Dimension

Code for *Understanding Scaling Laws with Statistical and Approximation Theory for Transformer Neural Networks on Intrinsically Low-dimensional Data*

## Installation

Assumes `python >= 3.9`

1. Install torch

2. `pip install -r requirements.txt`

## Usage

For model training: 

```bash
srun torchrun --standalone --nproc_per_node=4 train.py --dataset=openwebtext --always_save_checkpoint=True
```

For intrinsic dimension estimation:

```bash
python embeddings.py \
--model_path PATH_TO_SAVED_CHECKPOINT \
--model_mode oai \
--tokenizer_type oai \
--dataset_path PATH_TO_DATASET \
--split validation \
--dataset_mode hf \
--context_len 1024 \
--dataset_upper 4007 \
--num_dataset_subsample 4000 \
--shuffle_embeddings_per_sample \
--max_embeddings_per_sample 32 \
--shuffle_embeddings \
--max_embeddings 1000000
```

## Acknowledgements

Many thanks to the [nanoGPT project](https://github.com/karpathy/nanoGPT) project for their model training code.