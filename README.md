
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

## TensorBoard logging

Both the GPT pretraining script (`train.py`) and the Llama fine-tuning script (`train_llama.py`) can emit rich TensorBoard traces alongside checkpoints. Logging is disabled by default so existing workflows stay unchanged, but you can turn it on either from the command line or inside a config file.

### Enabling logging

- Pass `--tensorboard_log=True` when launching a run. The default log directory is `runs`, mirroring PyTorch's examples.
- Optionally set `--tensorboard_run_name=my_experiment` to make the run folder deterministic. When unset, the scripts fall back to the W&B run name (if provided) or a timestamped name.
- To change the base directory, override `--tensorboard_log_dir=/path/to/logs`.
- The flags are also available inside configuration files; simply assign `tensorboard_log = True` (and related options) in the config you pass to the training script.

Examples:

```bash
python train.py config/train_shakespeare_char.py \
    --tensorboard_log=True --tensorboard_run_name=shakespeare_debug

torchrun --standalone --nproc_per_node=4 train_llama.py config/finetune_llama3_lora.py \
    --tensorboard_log=True --tensorboard_run_name=llama_lora_experiment
```

When running distributed jobs, only the rank-0 process writes events, so you can point TensorBoard at a single directory even for multi-GPU training.

### Viewing dashboards

Launch TensorBoard and point it at the directory you chose above:

```bash
tensorboard --logdir runs
```

TensorBoard will default to serving on <http://localhost:6006>. Forward that port if you are on a remote machine.

### Metrics that are recorded

In addition to the standard loss curves, the logging now includes:

- Validation perplexity and the ratio between validation and training loss to highlight overfitting trends.
- The best validation loss reached so far for quick comparisons across runs.
- Step-level throughput (tokens/second and sequences/second) and the cumulative number of tokens processed.
- Mixed-precision diagnostics such as the gradient scaler value, gradient norm (clipped or unclipped), and the overall parameter norm of trainable weights.
- Hardware-oriented signals like model flop utilization (MFU) in `train.py`, along with per-iteration timing.

All scalars use iteration counts for the horizontal axis so you can align GPT and Llama runs easily when comparing dashboards.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!

## Llama 3.1 & 3.2 fine-tuning

Support for Meta's [Llama 3.1](https://huggingface.co/meta-llama) and [Llama 3.2](https://huggingface.co/meta-llama) models has been added via the `llama_model.py` and `train_llama.py` scripts. Pretrained checkpoints (1B, 8B, or 70B) are loaded from Hugging Face and LoRA or SingLoRA adapters can be applied for efficient fine-tuning. The adapters swap in for every linear layer in the Hugging Face model and are zero-initialized, so wrapping a checkpoint leaves its initial forward pass unchanged.

Example usage:

```sh
# Llama 3.2-1B (fastest, lowest memory)
python train_llama.py config/finetune_llama32_lora.py

# Llama 3.1-8B
python train_llama.py config/finetune_llama3_lora.py

# Llama 3.1-70B
python train_llama.py --config=config/finetune_llama3_lora.py --init_from=meta-llama/Llama-3.1-70B
```

**Model Size Comparison:**
- **Llama-3.2-1B**: ~1.2B parameters, fits on any GPU with 8GB+ VRAM
- **Llama-3.1-8B**: ~8B parameters, requires 24GB+ VRAM for LoRA training
- **Llama-3.1-70B**: ~70B parameters, requires significant resources

### Instruction-tuning datasets

Several supervised fine-tuning (SFT) corpora can be converted into the
binary format expected by `train_llama.py` via the helper script
`data/sft/prepare.py`. It downloads a dataset, formats each
instruction/response pair with the tokenizer's chat template and writes
`train.bin`/`val.bin` into `data/<dataset>/`, so the dataset can be
referenced directly via `--dataset <dataset>` when calling
`train_llama.py`.

Example for the Tulu personas dataset:

```sh
python data/sft/prepare.py --dataset tulu
python train_llama.py config/finetune_llama3_sft_qlora.py
```

### Supervised Fine-Tuning (SFT) with LoRA Variants

All LoRA variants support supervised fine-tuning for instruction-following and chat capabilities:

#### Available LoRA Variants for SFT:

- **LoRA**: Standard parameter-efficient fine-tuning (~0.5M trainable parameters)
- **SingleLoRA**: Single-matrix variant with ~50% fewer parameters (~0.25M trainable parameters)
- **QLoRA**: 4-bit quantized LoRA with ~95% memory reduction (~0.5M trainable parameters)
- **QSingleLoRA**: Quantized SingleLoRA with maximum memory efficiency (~0.25M trainable parameters)

#### SFT Configuration Files:

**Llama-3.2-1B (Recommended for fast experimentation):**
- `config/finetune_llama32_sft_lora.py` - Standard LoRA for SFT
- `config/finetune_llama32_sft_singlora.py` - SingleLoRA for SFT
- `config/finetune_llama32_sft_qlora.py` - QLoRA for SFT (most memory efficient)
- `config/finetune_llama32_sft_qsinglora.py` - QSingleLoRA for SFT (maximum efficiency)

**Llama-3.1-8B:**
- `config/finetune_llama3_sft_lora.py` - Standard LoRA for SFT
- `config/finetune_llama3_sft_singlora.py` - SingleLoRA for SFT
- `config/finetune_llama3_sft_qlora.py` - QLoRA for SFT
- `config/finetune_llama3_sft_qsinglora.py` - QSingleLoRA for SFT

#### Example SFT Commands:

```bash
# Llama-3.2-1B SFT (Fastest, lowest memory)
python train_llama.py config/finetune_llama32_sft_lora.py
python train_llama.py config/finetune_llama32_sft_singlora.py
python train_llama.py config/finetune_llama32_sft_qlora.py       # Recommended
python train_llama.py config/finetune_llama32_sft_qsinglora.py   # Most efficient

# Llama-3.1-8B SFT
python train_llama.py config/finetune_llama3_sft_lora.py
python train_llama.py config/finetune_llama3_sft_singlora.py
python train_llama.py config/finetune_llama3_sft_qlora.py
python train_llama.py config/finetune_llama3_sft_qsinglora.py
```

#### SFT Hyperparameters:

**Llama-3.2-1B (Higher batch sizes due to smaller model):**

| Parameter | LoRA | SingleLoRA | QLoRA | QSingleLoRA |
|-----------|------|------------|-------|-------------|
| `batch_size` | 8 | 8 | 16 | 16 |
| `learning_rate` | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| `lora_r` | 16 | - | - | - |
| `singlora_r` | - | 16 | - | - |
| `qlora_r` | - | - | 16 | - |
| `qsinglora_r` | - | - | - | 16 |
| `alpha` | 32 | 32 | 32 | 32 |
| `dropout` | 0.05 | 0.05 | 0.05 | 0.05 |
| `max_iters` | 2000 | 2000 | 2000 | 2000 |

**Llama-3.1-8B (Lower batch sizes due to larger model):**

| Parameter | LoRA | SingleLoRA | QLoRA | QSingleLoRA |
|-----------|------|------------|-------|-------------|
| `batch_size` | 4 | 4 | 2 | 2 |
| `learning_rate` | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| `lora_r` | 16 | - | - | - |
| `singlora_r` | - | 16 | - | - |
| `qlora_r` | - | - | 16 | - |
| `qsinglora_r` | - | - | - | 16 |
| `alpha` | 32 | 32 | 32 | 32 |
| `dropout` | 0.05 | 0.05 | 0.05 | 0.05 |
| `max_iters` | 2000 | 2000 | 2000 | 2000 |

### Dataset Selection for SFT

#### Supported Datasets:

The following instruction-tuning datasets are supported via `data/sft/prepare.py`:

- **`tulu`**: AllenAI Tulu 3 SFT Personas Instruction Following dataset
- **`ifeval`**: IFEval-like instruction following evaluation data
- **`autoif`**: AutoIF instruct dataset with 61k examples and functions

#### Preparing SFT Datasets:

```bash
# Prepare Tulu dataset (default)
python data/sft/prepare.py --dataset tulu

# Prepare IFEval dataset
python data/sft/prepare.py --dataset ifeval

# Prepare AutoIF dataset
python data/sft/prepare.py --dataset autoif

# Custom validation split (default 2%)
python data/sft/prepare.py --dataset tulu --val_split 0.05
```

#### Selecting Datasets in Training:

You can specify the dataset in several ways:

1. **Via config file** (recommended):
   ```python
   # In config/finetune_llama3_sft_qlora.py
   dataset = 'tulu'  # or 'ifeval' or 'autoif'
   ```

2. **Via command line override**:
   ```bash
   python train_llama.py config/finetune_llama3_sft_qlora.py --dataset=ifeval
   ```

3. **Custom dataset path** (if you prepare your own):
   ```bash
   python train_llama.py config/finetune_llama3_sft_qlora.py --dataset=/path/to/your/dataset
   ```

#### Dataset Details:

| Dataset | Size | Format | Best For |
|---------|------|--------|----------|
| `tulu` | ~50k examples | Chat messages with personas | General instruction following |
| `ifeval` | ~500 examples | Instruction-response pairs | Evaluation and testing |
| `autoif` | 61k examples | Chat messages with functions | Function calling and tools |

#### Memory Requirements by Model & LoRA Variant:

**Llama-3.2-1B (Very memory efficient):**

| Variant | Memory Usage | Trainable Params | Recommended For |
|---------|--------------|------------------|-----------------|
| LoRA | Very Low | ~0.5M | Any GPU (4GB+) |
| SingleLoRA | Very Low | ~0.25M | Any GPU (4GB+) |
| QLoRA | Minimal | ~0.5M | Any GPU (2GB+) |
| QSingleLoRA | Minimal | ~0.25M | Any GPU (2GB+) |

**Llama-3.1-8B:**

| Variant | Memory Usage | Trainable Params | Recommended For |
|---------|--------------|------------------|-----------------|
| LoRA | Medium | ~0.5M | 24GB+ GPUs |
| SingleLoRA | Medium-Low | ~0.25M | 16GB+ GPUs |
| QLoRA | Low | ~0.5M | 12GB+ GPUs |
| QSingleLoRA | Lowest | ~0.25M | 8GB+ GPUs |

#### Multi-GPU SFT Training:

All variants support distributed training:

```bash
# 8 GPUs with QLoRA (most efficient)
torchrun --standalone --nproc_per_node=8 train_llama.py config/finetune_llama3_sft_qlora.py

# Single GPU with LoRA
torchrun --standalone --nproc_per_node=1 train_llama.py config/finetune_llama3_sft_lora.py
```

#### SFT Best Practices:

1. **Start with QLoRA** for memory efficiency on consumer GPUs
2. **Use smaller learning rates** (2e-5) compared to pre-training
3. **Monitor validation loss** regularly with `eval_interval=500`
4. **Use appropriate batch sizes** based on your GPU memory
5. **Enable gradient clipping** (`grad_clip=1.0`) for stability
6. **Use bfloat16** for better numerical stability
7. **Consider early stopping** based on validation performance

#### Expected Training Times:

**Llama-3.2-1B (Much faster due to smaller model):**

| Variant | 8xA100 40GB | Single RTX 4090 | Single RTX 3090 |
|---------|--------------|-----------------|-----------------|
| LoRA | ~30 min | ~2 hours | ~3 hours |
| SingleLoRA | ~25 min | ~1.5 hours | ~2.5 hours |
| QLoRA | ~35 min | ~1.5 hours | ~2.5 hours |
| QSingleLoRA | ~30 min | ~1.2 hours | ~2 hours |

**Llama-3.1-8B:**

| Variant | 8xA100 40GB | Single RTX 4090 | Single RTX 3090 |
|---------|--------------|-----------------|-----------------|
| LoRA | ~4 hours | ~12 hours | ~18 hours |
| SingleLoRA | ~3.5 hours | ~10 hours | ~15 hours |
| QLoRA | ~5 hours | ~8 hours | ~12 hours |
| QSingleLoRA | ~4.5 hours | ~7 hours | ~10 hours |

*Times are estimates for 2000 iterations on Tulu dataset
