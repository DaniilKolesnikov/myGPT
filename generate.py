import torch
import argparse
import tiktoken
import os
from GPT import GPT, block_size, device

class CharTokenizer:
    def __init__(self, text_sample=None):
        if text_sample:
            # Create vocabulary from sample text
            chars = sorted(list(set(text_sample)))
            self.vocab = {ch: i for i, ch in enumerate(chars)}
            self.inverse_vocab = {i: ch for i, ch in enumerate(chars)}
            self.n_vocab = len(self.vocab)
        else:
            # Create a basic ASCII vocabulary as fallback
            chars = [chr(i) for i in range(32, 127)]  # Basic printable ASCII
            self.vocab = {ch: i for i, ch in enumerate(chars)}
            self.inverse_vocab = {i: ch for i, ch in enumerate(chars)}
            self.n_vocab = len(self.vocab)
    
    def encode(self, text):
        return [self.vocab.get(ch, self.n_vocab - 1) for ch in text]
    
    def decode(self, tokens):
        return ''.join([self.inverse_vocab.get(token, '?') for token in tokens])

def main():
    parser = argparse.ArgumentParser(description='Generate text from a trained GPT model')
    parser.add_argument('--checkpoint', type=str, default='model_checkpoints/checkpoint_4500.pth', 
                        help='Path to the model checkpoint')
    parser.add_argument('--prompt', type=str, default='', 
                        help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=500, 
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, 
                        help='Temperature for sampling (higher = more random)')
    parser.add_argument('--encoding', type=str, default='tiktoken', choices=['char', 'tiktoken'],
                        help='Tokenizer to use: char (character-level) or tiktoken')
    parser.add_argument('--sample_file', type=str, default=None,
                        help='Sample text file to build character vocabulary (for char encoding)')
    
    args = parser.parse_args()
    
    # Initialize tokenizer based on selection
    if args.encoding == 'char':
        # For character encoding, we need a sample text to build vocabulary
        if args.sample_file and os.path.exists(args.sample_file):
            with open(args.sample_file, 'r', encoding='utf-8') as f:
                sample_text = f.read()
            enc = CharTokenizer(sample_text)
            print(f"Using character-level encoding with vocabulary from {args.sample_file}")
        else:
            # Default to basic ASCII vocabulary
            enc = CharTokenizer()
            print("Using character-level encoding with basic ASCII vocabulary")
            if args.sample_file:
                print(f"Warning: Sample file {args.sample_file} not found, using fallback vocabulary")
    else:  # tiktoken
        enc = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's tokenizer
        print("Using tiktoken encoding (cl100k_base)")
    
    # Get vocabulary size for model initialization
    vocab_size = enc.n_vocab
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    model = GPT(vocab_size=vocab_size)
    
    # Load model checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoints = sorted(os.listdir("model_checkpoints"), 
                           key=lambda x: int(x.split('_')[1].split('.')[0]) if x.startswith('checkpoint_') else 0)
        for ckpt in checkpoints:
            print(f"  - model_checkpoints/{ckpt}")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    
    # Encode prompt
    if args.prompt:
        prompt_tokens = enc.encode(args.prompt)
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        prompt_tokens = prompt_tokens.unsqueeze(0)  # Add batch dimension (1, seq_len)
        print(f"Prompt: \"{args.prompt}\"")
    else:
        # Start with an empty context if no prompt
        prompt_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
        print("No prompt provided, starting with an empty context")
    
    # Generate text
    print("\nGenerating text...")
    with torch.no_grad():
        output_tokens = model.generate(
            prompt_tokens, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
    
    # Decode the output tokens
    generated_text = enc.decode(output_tokens[0].tolist())
    
    # Print the generated text
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(generated_text)
    print("="*50)

if __name__ == "__main__":
    main()