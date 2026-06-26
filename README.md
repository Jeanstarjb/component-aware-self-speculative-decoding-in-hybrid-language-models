# Component-Aware Self-Speculative Decoding in Hybrid Language Models

### Authors: Hector Borobia, Elies Seguí-Mas, Guillermina Tormo-Carbó  
**Paper URL:** [Read the full paper on arXiv](https://arxiv.org/pdf/2605.01106v1)

---

## Overview

### What is Speculative Decoding?
Speculative decoding is a technique that accelerates the process of generating text in language models. Instead of deciding each token sequentially, the model drafts multiple candidate tokens in parallel and validates them for coherence and relevance. This approach enables faster and more efficient text generation.

### What’s New in This Project?
This repository presents **component-aware self-speculative decoding**, a novel method that leverages the internal components of hybrid language models for speculative decoding. Unlike traditional approaches that rely on external tools or adjustments, this method utilizes the model's inherent architecture, offering faster inference without additional computational cost.

---

## Key Features

The Python/PyTorch code provided in this repository enables the following:

1. **Accelerated Inference**  
   Implements speculative decoding techniques to speed up text generation in hybrid language models.  

2. **Performance Evaluation**  
   Benchmarks speculative decoding efficiency for hybrid models (e.g., Falcon-H1 and Qwen3.5) and compares them to standard Transformer-based architectures.

3. **Architectural Analysis**  
   Investigates how the internal design of hybrid models influences speculative decoding performance, providing predictive insights without requiring extensive experiments.

---

## Core Concepts

Hybrid language models combine multiple architectural components, such as linear attention layers and Transformer blocks, to enhance text generation capabilities. This project focuses on understanding how these components interact and affect speculative decoding efficiency. 

For example:
- **Parallel hybrids** (e.g., Falcon-H1) demonstrate high acceptance rates for candidate tokens (up to 68%).  
- **Sequential hybrids** (e.g., Qwen3.5) struggle with low acceptance rates (around 3.8%).  

This disparity highlights the importance of architectural integration in determining speculative decoding success.

---

## Results and Insights

### Key Findings
1. **Scale Independence**  
   The performance of Falcon-H1 remains consistent across different model sizes—from 0.5 billion to 3 billion parameters.

2. **Predictive Indicators**  
   An ablation study reveals that monitoring perplexity degradation can predict the success of speculative decoding, offering a practical way to assess hybrid architectures.

3. **Efficiency Metrics**  
   Models with better internal composition achieve faster and more intelligent speculative decoding, emphasizing the importance of architectural design.

---

## Getting Started

Follow the steps below to use the code:

### Clone the Repository
```bash
git clone https://github.com/your-repo/component-aware-decoding.git
cd component-aware-decoding
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Speculative Decoding Experiments
To test speculative decoding on a hybrid model:
```bash
python decode_hybrid.py --model falcon-h1 --draft_length 2
```

### Visualize Results
Plot experimental outcomes using the built-in visualization tools:
```bash
python plot_results.py
```

---

## Why This Matters

This project is not just about faster text generation. It provides deeper insights into hybrid language model architectures, paving the way for:
- Development of more computationally efficient AI systems.  
- Enhanced understanding of speculative decoding mechanisms.  
- Practical applications in tasks requiring rapid and accurate autoregressive inference.

---

## Contribute and Collaborate

We welcome contributions and feedback! If you have questions, suggestions, or would like to report issues, please open an issue or submit a pull request. Together, let's innovate in hybrid language modeling and speculative decoding.

---