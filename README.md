# MMN 13 (Course 22969) - KV-Cache Inference Acceleration for GPT-2

Author: Amir Noiboar (ID 027480110)

## Project Overview

This project investigates inference-time acceleration in GPT2-style language models using Key-Value (KV) caching.
The goal is to demonstrate how KV-cache reduces decoding complexity from quadratic O(n²) to linear O(n) with respect
to prompt length, while introducing an additional memory cost.

## Motivation
The work was motivated by the Mamba architecture, which achieves linear-time sequence modeling,
and explores whether similar efficiency gains can be obtained within the standard Transformer framework.

## Main Contributions

Implementation of KV-cache support in a GPT-2 model based on the
LLMs-from-scratch repository by Sebastian Raschka.

Verification of identical outputs between:

Regular autoregressive decoding

KV-cache-accelerated decoding

Benchmarking inference speed across:

Multiple prompt lengths

Multiple model sizes (GPT-2 Small, Medium, Large)

Empirical demonstration of:

Stable decoding throughput with KV-cache

Quadratic slowdown without KV-cache

The memory–speed trade-off introduced by caching

## How to Run

Create a Python environment with PyTorch + CUDA.

Open the notebook MMN_13_course_22969_Amir_Noiboar.ipynb and run all cells to:

Load pretrained GPT-2 weights

Verify identical generation

Execute benchmarking

Produce plots

## Key Result

KV-cache provides significant decoding speedup for:

Long prompts

Larger GPT-2 models

while requiring additional linear memory to store cached keys and values.
 