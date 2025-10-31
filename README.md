PyTorch implementation of Stable Diffusion from scratch

This repository inspired by https://github.com/hkproj/pytorch-stable-diffusion, but my contribution is to add annotations for beginner to learn how the stable diffusion works even if it's a little bit hard to read.

## Download weights and tokenizer files:
- Download vocab.json and merges.txt from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer and save them in the data folder
- Download v1-5-pruned-emaonly.ckpt from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main and save it in the data folder

## Tested fine-tuned models:
Just download the ckpt file from any fine-tuned SD (up to v1.5).

1.InkPunk Diffusion: https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main
2.Illustration Diffusion (Hollie Mengert): https://huggingface.co/ogkalu/Illustration-Diffusion/tree/main
