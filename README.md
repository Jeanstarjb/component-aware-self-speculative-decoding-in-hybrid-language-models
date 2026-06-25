# 🚀 Component-Aware Self-Speculative Decoding in Hybrid Language Models

### Authors: Hector Borobia, Elies Seguí-Mas, Guillermina Tormo-Carbó  
📄 **Paper URL:** [Read the full paper on arXiv](https://arxiv.org/pdf/2605.01106v1)

---

## 🌟 Overview

✨ **What is speculative decoding?**  
Imagine you’re writing a sentence. Instead of waiting to decide each word one by one, you draft some ideas quickly, then check if they make sense. Speculative decoding speeds up this process for language models—producing faster, smarter results!  

🔍 **What’s new here?**  
This project introduces **component-aware self-speculative decoding**, a groundbreaking method that leverages the internal architecture of hybrid language models. Unlike traditional approaches that rely on external tools, this method uses the model’s own components to draft candidate tokens internally—at zero extra cost!  

---

## 🤖 Key Features of the Code

This repository includes Python/PyTorch implementations to:  
1. **Accelerate inference** using speculative decoding techniques.  
2. **Evaluate hybrid language models** like Falcon-H1 and Qwen3.5, alongside pure Transformer baselines.  
3. **Analyze architectural patterns** to predict speculative decoding efficiency—without needing to run decoding experiments!  

---

## 🧠 What Does This Mean?

Hybrid language models are like multi-talented teams: they combine different components (like linear attention layers and Transformers) to process language better. This project digs deep into these models' inner workings to figure out:  
- How their unique architectures affect speculative decoding.  
- Why some hybrids perform much better than others.  

For example:  
- **Parallel hybrids** (Falcon-H1) achieve high acceptance rates of candidate drafts—up to 68%! 🚀  
- **Sequential hybrids** (Qwen3.5), on the other hand, struggle with only 3.8% acceptance. 😅  

The difference? It’s all about how each architecture integrates its components.  

---

## 📊 Results Breakdown

Here’s the magic in numbers:  
- **Scale doesn't matter!** Falcon-H1 performs consistently at both small (0.5B) and large (3B) scales.  
- **Predictive insights!** A clever ablation study shows that perplexity degradation can predict speculative decoding success.  

To sum it up: the better the model’s internal composition, the faster and smarter its speculative decoding becomes.  

---

## 🛠️ How to Use the Code

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/component-aware-decoding.git
   cd component-aware-decoding
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the speculative decoding experiments on hybrid models:  
   ```bash
   python decode_hybrid.py --model falcon-h1 --draft_length 2
   ```

4. Visualize your results with built-in plotting tools:  
   ```bash
   python plot_results.py
   ```

---

## ✨ Why It’s Exciting

This isn’t just a faster way to decode—it’s a smarter way to understand hybrid language models and their architectures. Whether you're a researcher or a developer, this project opens doors to:  
- Efficient AI systems with minimal computational cost.  
- Insights into next-gen hybrid architectures.  
- Applications in real-world tasks needing lightning-fast autoregressive inference.  

---

## 📬 Get Involved  

Have questions or ideas? Reach out via issues or contribute to the repository! Let’s decode the future together. 🚀  

---