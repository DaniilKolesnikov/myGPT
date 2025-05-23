from GPT import *
import argparse
import tiktoken
import torch.cuda.amp as amp  # Import AMP for mixed precision training
import os
import glob

torch.manual_seed(1337)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'

# Add command line arguments
parser = argparse.ArgumentParser(description='Train a GPT model with character-level or tiktoken tokenization')
parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'tiktoken'], 
                   help='Tokenizer to use: char for character-level, tiktoken for OpenAI tokenizer')
parser.add_argument('--tiktoken_model', type=str, default='cl100k_base', 
                   help='Tiktoken model to use, default is cl100k_base (used in GPT-4)')
parser.add_argument('--mixed_precision', action='store_true',
                   help='Enable mixed precision training (fp16/fp32) for faster training on supported GPUs')
parser.add_argument('--batch_size', type=int, default=batch_size,
                   help=f'Batch size for training (default: {batch_size})')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                   help='Number of steps to accumulate gradients before updating weights (default: 1)')
parser.add_argument('--checkpoint_path', type=str, default='model_checkpoints',
                   help='Path to save checkpoints during training')
parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                   help='Path to a checkpoint to resume training from')
parser.add_argument('--keep_n_checkpoints', type=int, default=3,
                   help='Number of most recent checkpoints to keep (default: 3)')
parser.add_argument('--train_text', type=str, default='jokes.txt',
                   help='Path to the training text file (default: jokes.txt)')
parser.add_argument('--lr_scheduler', type=str, default='constant', choices=['constant', 'cosine', 'linear'],
                   help='Learning rate scheduler: constant, cosine, or linear decay')
parser.add_argument('--lr_warmup_steps', type=int, default=0,
                   help='Number of warmup steps for learning rate')
parser.add_argument('--min_lr', type=float, default=1e-5,
                   help='Minimum learning rate for schedulers')
args = parser.parse_args()

# Update batch size from arguments
batch_size = args.batch_size

# Create directories for checkpoints
os.makedirs(args.checkpoint_path, exist_ok=True)

# Create GradScaler for mixed precision training
scaler = torch.GradScaler(enabled=args.mixed_precision)

if args.train_text == 'mixed':
    with open('jokes.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    with open('poems.txt', 'r', encoding='utf-8') as f:
        text += f.read()
else:
    with open(args.train_text, 'r', encoding='utf-8') as f:
        text = f.read()

# Initialize tokenizer based on the argument
if args.tokenizer == 'char':
    # Character-level tokenization (original implementation)
    characters = sorted(list(set(text)))
    vocab_size = len(characters)

    stoi = {ch: i for i, ch in enumerate(characters)}
    itos = {i: ch for i, ch in enumerate(characters)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return ''.join([itos[i] for i in l])
else:
    # Tiktoken tokenization
    enc = tiktoken.get_encoding(args.tiktoken_model)
    vocab_size = enc.n_vocab

    def encode(s):
        return enc.encode(s)

    def decode(l):
        return enc.decode(l)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Use the same precision setting for evaluation
            with torch.autocast(device_type='cuda', enabled=args.mixed_precision):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(vocab_size)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Create learning rate scheduler based on argument
if args.lr_scheduler == 'cosine':
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=max_iters - args.lr_warmup_steps, 
        eta_min=args.min_lr
    )
elif args.lr_scheduler == 'linear':
    # Linear decay
    lambda1 = lambda epoch: max(args.min_lr / learning_rate, 
                               1.0 - (epoch - args.lr_warmup_steps) / (max_iters - args.lr_warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
else:
    # Constant learning rate (no scheduling)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

# Resume from checkpoint if specified
if args.resume_from_checkpoint:
    model.load_state_dict(torch.load(args.resume_from_checkpoint))
    print(f"Resumed training from checkpoint: {args.resume_from_checkpoint}")

#--------------------------------------------------
# Training loop

for iter in tqdm(range(max_iters)):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr: {current_lr:.7f}")
        checkpoint_filename = os.path.join(args.checkpoint_path, f'checkpoint_{iter}.pth')
        torch.save(model.state_dict(), checkpoint_filename)
        print(f"Checkpoint saved as {checkpoint_filename}")
        
        # Rotate checkpoints - keep only the most recent N
        if args.keep_n_checkpoints > 0:
            checkpoints = glob.glob(os.path.join(args.checkpoint_path, 'checkpoint_*.pth'))
            # Extract iteration numbers and sort numerically
            checkpoints = sorted(checkpoints, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            if len(checkpoints) > args.keep_n_checkpoints:
                for checkpoint_to_delete in checkpoints[:-args.keep_n_checkpoints]:
                    os.remove(checkpoint_to_delete)
                    print(f"Removed old checkpoint: {checkpoint_to_delete}")

    # Zero gradients
    optimizer.zero_grad()
    
    # Accumulate gradients over multiple steps
    for _ in range(args.gradient_accumulation_steps):
        # Get batch data
        X, Y = get_batch('train')
        
        # Forward pass with automatic mixed precision (when enabled)
        with torch.autocast(device_type='cuda', enabled=args.mixed_precision):
            logits, loss = model(X, Y)
        
        # Backward pass with gradient scaling (for mixed precision)
        scaler.scale(loss).backward()
    
    # Unscale gradients and perform optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # Step the scheduler
    scheduler.step()

# Save the model with appropriate name based on tokenizer
model_filename = 'model_tiktoken.pth' if args.tokenizer == 'tiktoken' else 'model.pth'
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")