# TAAC2026 HyFormer Baseline

This repository contains a self-contained PyTorch baseline for TAAC2026 PCVR
prediction. The model is a HyFormer-style architecture that combines
non-sequence user/item features with multiple historical behavior sequences.

## Overview

The baseline predicts whether the current candidate item will convert after a
click. It reads offline Parquet feature files, builds structured batches, and
trains a binary classifier with BCE or Focal Loss.

Main components:

- `train.py`: training entry point, argument parsing, data/model/trainer wiring.
- `dataset.py`: offline Parquet reader and batch construction.
- `model.py`: `PCVRHyFormer` model, sequence encoders, NS tokenizers, HyFormer blocks.
- `trainer.py`: training loop, validation, checkpointing, early stopping.
- `utils.py`: logging, random seed setup, early stopping, focal loss.
- `ns_groups.json`: example NS feature grouping config.
- `run.sh`: example launch script.

## Data Layout

The code expects an offline data directory containing Parquet files and a schema:

```text
data_dir/
  schema.json
  *.parquet
```

The default schema path is:

```text
<data_dir>/schema.json
```

The dataset reader maps labels as:

```text
positive = label_type == 2
negative = all other label_type values
```

The Parquet reader is implemented as an `IterableDataset`. It reads Parquet Row
Groups with `pyarrow`, converts each Arrow batch into a tensor dictionary, and
passes already-batched samples to PyTorch `DataLoader(batch_size=None)`.

## Batch Format

Each yielded batch is a dictionary containing non-sequence features, sequence
features, sequence lengths, time buckets, and labels:

```python
{
    "user_int_feats": Tensor,        # [B, user_int_total_dim]
    "user_dense_feats": Tensor,      # [B, user_dense_total_dim]
    "item_int_feats": Tensor,        # [B, item_int_total_dim]
    "item_dense_feats": Tensor,      # [B, item_dense_total_dim], often [B, 0]
    "label": Tensor,                 # [B]
    "timestamp": Tensor,             # [B]
    "user_id": list,                 # length B
    "_seq_domains": list,            # e.g. ["seq_a", "seq_b", "seq_c", "seq_d"]
    "seq_a": Tensor,                 # [B, S_a, L_a]
    "seq_a_len": Tensor,             # [B]
    "seq_a_time_bucket": Tensor,     # [B, L_a]
    ...
}
```

For sequence tensors, `S` is the number of side-info fields in that sequence
domain, and `L` is the padded/truncated sequence length. A single historical
event corresponds to one time position across all `S` fields.

## Model

`PCVRHyFormer` builds two kinds of tokens:

- S tokens: sequence tokens. For each sequence domain, each field is embedded,
  embeddings at the same time position are concatenated, and the result is
  projected to `d_model`, producing `[B, L, d_model]`.
- NS tokens: non-sequence tokens. User/item sparse features are tokenized with
  either `group` or `rankmixer`; dense features are projected into additional
  tokens when present.

Each HyFormer block performs:

1. Sequence evolution: each sequence domain is encoded independently.
2. Query decoding: per-domain query tokens attend to their corresponding sequence.
3. Query boosting: decoded query tokens and NS tokens are mixed by `RankMixerBlock`.

The final query tokens are concatenated, projected, and fed into a binary
classifier.

## Target-Aware Query

This branch adds an optional target-aware query generator:

```bash
--target_aware_query
```

When enabled, the query generator explicitly conditions each per-sequence query
on the current candidate item representation:

```text
old: query = MLP([NS_flat, seq_pooled])
new: query = MLP([NS_flat, item_summary, seq_pooled])
```

`item_summary` is the mean-pooled item-side NS token slice. The flag is disabled
by default so the original baseline can still be reproduced directly.

## Sparse Re-Init

The baseline includes an optional high-cardinality embedding re-initialization
strategy controlled by:

```bash
--reinit_cardinality_threshold
--reinit_sparse_after_epoch
```

Embedding tables whose vocabulary size exceeds `reinit_cardinality_threshold`
are re-initialized after the configured epoch. With the default
`reinit_cardinality_threshold=0`, every embedding table with `vocab_size > 0`
is re-initialized. This keeps the original cold-restart behavior, which can help
reduce overfitting on high-cardinality sparse IDs.

## Running

Set the required paths:

```bash
export TRAIN_DATA_PATH=/path/to/data_dir
export TRAIN_CKPT_PATH=/path/to/checkpoints
export TRAIN_LOG_PATH=/path/to/logs
export TRAIN_TF_EVENTS_PATH=/path/to/tensorboard_events
```

Run the default script:

```bash
bash run.sh
```

Or run directly:

```bash
python train.py \
  --batch_size 256 \
  --num_epochs 999 \
  --patience 5 \
  --loss_type bce
```

Run with target-aware query:

```bash
python train.py \
  --target_aware_query \
  --batch_size 256 \
  --loss_type bce
```

## Important Hyperparameters

- `--ns_tokenizer_type`: `rankmixer` or `group`.
- `--user_ns_tokens`, `--item_ns_tokens`: token counts for rankmixer NS tokenizer.
- `--seq_max_lens`: per-domain sequence truncation, e.g. `seq_a:256,seq_b:256`.
- `--seq_encoder_type`: `swiglu`, `transformer`, or `longer`.
- `--num_queries`: number of query tokens generated per sequence domain.
- `--rank_mixer_mode`: `full`, `ffn_only`, or `none`.
- `--loss_type`: `bce` or `focal`.
- `--reinit_cardinality_threshold`: embeddings with larger vocab size are
  re-initialized; `0` resets every embedding with `vocab_size > 0`.

## Suggested Ablations

Useful first experiments:

- Baseline vs `--target_aware_query`.
- `ns_tokenizer_type=rankmixer` vs `group`.
- Different `user_ns_tokens` and `item_ns_tokens`.
- Different `seq_max_lens` per sequence domain.
- `loss_type=bce` vs `focal`, after checking the positive sample ratio.
- `seq_encoder_type=transformer` vs `longer` for long sequence domains.

## Checkpoints

The trainer monitors validation AUC and saves the current best model under:

```text
<TRAIN_CKPT_PATH>/global_step*.best_model/
```

Each checkpoint directory includes:

- `model.pt`
- `schema.json`
- `ns_groups.json` when available
- `train_config.json`
