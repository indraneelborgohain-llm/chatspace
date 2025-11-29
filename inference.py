import torch
from torch.nn import functional as F
import os
import argparse

from architecture.tokenizer import get_tokenizer
from architecture.gptoss import Transformer, ModelConfig




context_len=8192
tokenizer= get_tokenizer()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())



def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    
    idx = text_to_token_ids(prompt,tokenizer).to(device)
    # Generate
    for _ in range(max_tokens):
        idx_cond = idx[-context_len:]
        with torch.inference_mode():
            logits= model(idx_cond)
        logits = logits[-1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=0)

    
    
       
    
    # Decode and return
    result = token_ids_to_text(idx,tokenizer)
    return result


def load_model_and_generate(
    checkpoint_path,
    prompt="Once upon a time",
    max_tokens=200,
    temperature=0.8,
    top_k=200,
    device=None
):
    """
    Load a trained model from checkpoint and generate text.
    
    Args:
        checkpoint_path: Path to the checkpoint file (e.g., 'model/gptoss_best.pt')
        prompt: Text prompt to start generation
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        device: Device to use ('cuda:0', 'cpu', etc.). Auto-detect if None.
    
    Returns:
        Generated text string
    """
    # Auto-detect device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config_dict = checkpoint["config"]
        # Build ModelConfig from saved args
        vocab_size = 201088  # o200k_harmony
        
        # Determine model size from config
        if "model_size" in config_dict:
            model_size = config_dict["model_size"]
        else:
            # Try to infer from saved model architecture
            model_size = "toy"  # default
        
        # Build config based on model size
        from train import build_config
        cfg = build_config(model_size, vocab_size)
        
    else:
        # Default config if not saved
        print("Warning: Config not found in checkpoint, using default toy config")
        cfg = ModelConfig(
            vocab_size=201088,
            hidden_size=512,
            num_hidden_layers=6,
            head_dim=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
            experts_per_token=2,
            intermediate_size=512,
            sliding_window=64,
            initial_context_length=2048,
        )
    
    # Create model
    print(f"Creating model with config: {cfg.hidden_size}d, {cfg.num_hidden_layers} layers")
    model = Transformer(cfg, device=device)
    
    # Load weights
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "iter_num" in checkpoint:
            print(f"Loaded checkpoint from iteration {checkpoint['iter_num']}")
        if "best_val_loss" in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded successfully!")
    print(f"\nGenerating text with prompt: '{prompt}'")
    print("=" * 70)
    
    # Generate text
    generated = generate_text(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/gptoss_best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda:0, cpu, etc.)"
    )
    
    args = parser.parse_args()
    
    # Generate text
    generated_text = load_model_and_generate(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )
    
    print(generated_text)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
