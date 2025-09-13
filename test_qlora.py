#!/usr/bin/env python3
"""
Test script to demonstrate QLoRA and QSingleLoRA implementations.
"""

import torch
from model import GPT, GPTConfig

def test_qlora():
    """Test QLoRA implementation."""
    print("Testing QLoRA...")

    # Configure QLoRA
    config = GPTConfig(
        n_layer=2,  # Small model for testing
        n_head=4,
        n_embd=64,
        vocab_size=1000,
        block_size=128,
        qlora_r=8,
        qlora_alpha=16,
        qlora_dropout=0.0,
        qlora_bits=4,
        qlora_blocksize=32
    )

    model = GPT(config)

    # Create sample input
    batch_size, seq_len = 2, 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss = model(x)

    print(f"QLoRA test successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of parameters: {model.get_num_params()}")
    print()

def test_qsinglora():
    """Test QSingleLoRA implementation."""
    print("Testing QSingleLoRA...")

    # Configure QSingleLoRA
    config = GPTConfig(
        n_layer=2,  # Small model for testing
        n_head=4,
        n_embd=64,
        vocab_size=1000,
        block_size=128,
        qsinglora_r=8,
        qsinglora_alpha=16,
        qsinglora_dropout=0.0,
        qsinglora_bits=4,
        qsinglora_blocksize=32
    )

    model = GPT(config)

    # Create sample input
    batch_size, seq_len = 2, 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss = model(x)

    print(f"QSingleLoRA test successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of parameters: {model.get_num_params()}")
    print()

def test_mutual_exclusivity():
    """Test that only one LoRA variant can be enabled."""
    print("Testing mutual exclusivity...")

    try:
        # Try to enable both QLoRA and QSingleLoRA
        config = GPTConfig(
            n_layer=2,
            n_head=4,
            n_embd=64,
            vocab_size=1000,
            block_size=128,
            qlora_r=8,
            qsinglora_r=8  # This should fail
        )
        model = GPT(config)  # This should trigger the assertion
        print("ERROR: Should have failed mutual exclusivity check!")
    except AssertionError as e:
        print(f"✓ Mutual exclusivity check passed: {e}")

    try:
        # Try to enable both LoRA and QLoRA
        config = GPTConfig(
            n_layer=2,
            n_head=4,
            n_embd=64,
            vocab_size=1000,
            block_size=128,
            lora_r=8,
            qlora_r=8  # This should fail
        )
        model = GPT(config)  # This should trigger the assertion
        print("ERROR: Should have failed mutual exclusivity check!")
    except AssertionError as e:
        print(f"✓ Mutual exclusivity check passed: {e}")

if __name__ == "__main__":
    print("Testing QLoRA and QSingleLoRA implementations\n")

    test_qlora()
    test_qsinglora()
    test_mutual_exclusivity()

    print("All tests completed!")
