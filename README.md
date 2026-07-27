# CLIP in JAX/Flax

A clean, modular reimplementation of OpenAI's CLIP (Contrastive Language-Image Pretraining) model using JAX and the Flax `nnx` API. 

This repository provides a ground-up implementation of the Vision Transformer (ViT) and Text Transformer architectures, along with a custom weight-mapping utility to load pre-trained Hugging Face weights directly into the custom `nnx` state.

## Features

- **Flax `nnx` API:** Built using the stateful neural network API for Flax.
- **Modular Architecture:** Separate, easily hackable modules for the vision and text encoders.
- **Native Weight Conversion:** Maps official Hugging Face `.msgpack` weights to custom JAX arrays, handling shape reshaping and key mapping directly.

## Repository Structure

- `modeling.py`: The unified CLIP model connecting the vision and text towers.
- `text_encoder.py`: Text Transformer implementation with causal masking.
- `vision_encoder.py`: Vision Transformer implementation for image patch extraction.
- `conversion.py`: Core utility to map and load HF layer names/shapes into the `nnx` model state.
- `inference.py`: End-to-end inference script evaluating semantic similarity between an image and candidate text labels.
- `requirements.txt`: Project dependencies.

## Installation

1. **Clone the repository**:
   
   ```bash
   git clone https://github.com/yourusername/clip-flax.git
   cd clip-flax
   ```

2. Create a virtual environment and install dependencies:
   
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. **Download Weights:** Download the `flax_model.msgpack` file for `openai/clip-vit-base-patch32` from the Hugging Face Hub and place it in the root directory. 

## Usage

Run the inference script to test zero-shot image classification. The script automatically fetches a sample image of a dog from Unsplash and evaluates its semantic similarity against candidate text labels (`["a photo of a dog", "a photo of a cat", "a photo of a car"]`).

```bash
python inference.py
```

## Performance Note & Benchmarks

This repository is designed for **architectural transparency, modularity, and educational research** rather than raw production speed. 




[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-p4KWMzp__-RbnRVXaIJd0rCr_Oncoco?usp=sharing)

Because this custom Flax `nnx` implementation prioritizes explicit, readable mathematical operations over end-to-end black-box abstractions, it does not currently compete with the official Hugging Face implementation in terms of inference time.


For context, a provided benchmark notebook (`benchmark.ipynb`) comparing highly optimized production frameworks on an **NVIDIA T4 GPU** shows the baseline targets:

* **JAX (via Keras 3):** Using `keras_hub` with the JAX backend and JIT compilation, the model achieves a throughput of 201.05 samples/sec with an average batch latency of 159.13 ms. It requires a significant initial JIT compilation overhead (the first warmup iteration takes ~21.2 seconds).

* **PyTorch (via Hugging Face):** Using the official `transformers` implementation with `torch.compile`, the model achieves a throughput of 186.51 samples/sec with an average batch latency of 171.52 ms. The initial compilation overhead is lower, taking only 337.79 ms.
  
  

Expect the custom `nnx` implementation in this repository to have longer execution times than the highly optimized frameworks listed above for single-device inference. The primary value of this codebase is the ability to easily hack, inspect, and modify the multi-modal extraction pipelines directly in pure JAX. 



Additionally, a native JAX foundation provides a massive advantage for **custom pretraining**. By controlling the architecture from the ground up, you can seamlessly utilize JAX's native parallelization capabilities (such as `jax.pmap` and `jax.sharding`) to distribute training across multi-GPU setups or TPUs. This grants complete flexibility over the training loop, making it significantly easier to experiment with novel contrastive loss functions, custom masking strategies, or alternative multi-modal fusion techniques before exporting a final model.


