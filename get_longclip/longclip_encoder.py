import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'get_longclip'))
import longclip
sys.path.append('..')

device = "cuda" if torch.cuda.is_available() else "cpu"
vitl_model, vitl_preprocess = longclip.load("ckpt/longclip-L.pt", device=device)
vitl_model.eval()
vitL_encoder = vitl_model.encode_text_full

with torch.no_grad():
    def encode_prompt(
            prompt: str,
            device: Optional[torch.device] = None,
    ):

        # Define tokenizers and text encoders
        tokenizer = longclip.tokenize
        text_encoder = vitL_encoder

        # Process prompts
        text_inputs = tokenizer(prompt)
        text_input_ids = text_inputs
        untruncated_ids = tokenizer(prompt)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.decoder(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}."
            )

        prompt_embeds = text_encoder(text_input_ids.to(device))
        prompt_embeds = prompt_embeds.to(torch.float32)

        return prompt_embeds

# torch.Size([1, 248, 768])