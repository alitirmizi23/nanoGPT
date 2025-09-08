import os
import argparse
import pickle
from typing import List, Dict

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Mapping from short names to Hugging Face hub identifiers
HF_DATASETS = {
    "tulu": "allenai/tulu-3-sft-personas-instruction-following",
    "ifeval": "argilla/ifeval-like-data",
    "autoif": "Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs",
}


def build_messages(name: str, example: Dict) -> List[Dict[str, str]]:
    """Return a chat-style list of messages for different dataset formats."""
    if name == "tulu":
        messages = example["messages"]
    elif name == "ifeval":
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
    elif name == "autoif":
        messages = example["messages"]
        system = example.get("system")
        if system:
            messages = [{"role": "system", "content": system}] + messages
    else:
        raise ValueError(f"unknown dataset name: {name}")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT datasets for nanoGPT")
    parser.add_argument("--dataset", choices=HF_DATASETS.keys(), required=True,
                        help="which dataset to prepare")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                        help="tokenizer name")
    parser.add_argument("--val_split", type=float, default=0.02,
                        help="fraction of examples for validation")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(HF_DATASETS[args.dataset])
    split = dataset["train"].train_test_split(test_size=args.val_split,
                                              seed=2357, shuffle=True)
    split["val"] = split.pop("test")

    def tokenize(example: Dict) -> Dict:
        ids = tokenizer.apply_chat_template(
            build_messages(args.dataset, example),
            tokenize=True,
            add_generation_prompt=False,
        )
        ids.append(tokenizer.eos_token_id)
        return {"ids": ids, "len": len(ids)}

    remove_cols = dataset["train"].column_names
    tokenized = split.map(tokenize, remove_columns=remove_cols,
                          desc="tokenizing", num_proc=4)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(script_dir), args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    vocab_size = tokenizer.vocab_size
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32

    for part, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(out_dir, f"{part}.bin")
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        idx = 0
        for ids in tqdm(dset["ids"], desc=f"writing {filename}"):
            arr[idx:idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()

    meta = {"vocab_size": vocab_size}
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main()
