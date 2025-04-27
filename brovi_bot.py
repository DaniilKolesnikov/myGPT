import logging
import random
import argparse
import tiktoken
import torch.cuda.amp as amp
import gc
from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove, Update, InlineKeyboardButton, InlineKeyboardMarkup)
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters)

from GPT import *

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Add command line arguments
parser = argparse.ArgumentParser(description='Run a Telegram bot using a GPT model with character-level or tiktoken tokenization')
parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'tiktoken'], 
                   help='Tokenizer to use: char for character-level, tiktoken for OpenAI tokenizer')
parser.add_argument('--tiktoken_model', type=str, default='cl100k_base', 
                   help='Tiktoken model to use, default is cl100k_base (used in GPT-4)')
parser.add_argument('--mixed_precision', action='store_true',
                   help='Enable mixed precision for generation (fp16/fp32)')
parser.add_argument('--model_file', type=str, default='model.pth',
                   help='Path to the model file to load')
parser.add_argument('--memory_efficient', action='store_true',
                   help='Run in memory-efficient mode, with reduced context size')
args = parser.parse_args()

with open('jokes.txt', 'r') as f:
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

model = GPT(vocab_size)
model.to(device)

model.load_state_dict(torch.load(args.model_file))

# Create autocast context for mixed precision
autocast = amp.autocast(enabled=args.mixed_precision)

# For memory efficiency, potentially reduce block size for generation
generation_block_size = block_size // 2 if args.memory_efficient else block_size

# Print memory stats at startup
def print_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

print_gpu_memory()

# Handler for the /start command
async def start(update, context):
    await update.message.reply_text("привет. Я бот брови латышАИ. Я могу сгенерировать смехуечки. Напиши начало фразы которую хочешь продолжить или /generate чтобы я придумал сам.")

# Handler for the /generate command
async def generate(update, context):
    words_num = random.randint(2, 10)
    c = torch.zeros((1, 1), dtype=torch.long, device=device)
    with autocast:
        generated_text = decode(model.generate(c, 100, temperature=0.8)[0].tolist())
    # Split the generated text into words and limit to the specified number
    words = generated_text.split()[:words_num]
    await update.message.reply_text(" ".join(words))

# Handler for custom text messages (non-commands)
async def echo(update, context):
    token_num = len(encode(update.message.text))
    # Generate text based on the user's input
    inputed = torch.tensor(encode(update.message.text), dtype=torch.long, device=device)
    inputed = inputed.unsqueeze(0)
    with autocast:
        generated_text = decode(model.generate(
            inputed, 
            int(token_num * (random.random() + (2 if token_num > 10 else 4))),
            temperature=0.8
        )[0].tolist())
    # Split the generated text into words and limit to the specified number
    words = generated_text.split()[:-1]
    await update.message.reply_text(" ".join(words))

# Handler for the /joke command
async def joke(update, context):
    with autocast:
        generated_text = decode(model.generate(
            torch.zeros((1, 1), dtype=torch.long, device=device), 
            200,
            temperature=0.7  # Lower temperature for more coherent jokes
        )[0].tolist())
    selected_joke = generated_text.split('\n\n')[0]
    await update.message.reply_text(selected_joke)

# Main function to set up and run the bot
def main():
    # Replace 'YOUR_TOKEN_HERE' with the token from BotFather
    application = Application.builder().token("7649564423:AAGQPXmRb5vbHtH6m02xnXbzqDBYdvJBhrg").build()

    # Add handlers for commands and messages
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("generate", generate))
    application.add_handler(CommandHandler("complete", echo))
    application.add_handler(CommandHandler("joke", joke))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Start the bot
    application.run_polling()

# Run the script
if __name__ == "__main__":
    main()