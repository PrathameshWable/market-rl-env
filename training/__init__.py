"""Training-time utilities: prompts, rollouts, curriculum, SFT data gen.

This package is imported only by the training notebook and the SFT data
generation script. It is NOT imported by the runtime environment server,
so it can pull in heavy ML deps (transformers, trl, unsloth) without
bloating the HF Space image.
"""
