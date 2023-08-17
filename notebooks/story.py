# Data
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import seaborn as sns  # makes heatmap look better
from datasets import load_dataset
import sentencepiece as spm
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from collections import Counter
import json
import wandb


vocab_size = 7600  # english has 26 * 2 + punctuation

dataset = load_dataset("roneneldan/TinyStories")

train_dataset = dataset["train"][:50000]

validation_dataset = dataset["validation"][:50000]
text_data = [entry for entry in train_dataset["text"]]
validation_data = [entry for entry in validation_dataset["text"]]

text_data_str = "\n".join(text_data)
with open("temp.txt", "w", encoding="utf-8") as f:
    f.write(text_data_str)

spm.SentencePieceTrainer.train(
    f"--input=temp.txt --model_prefix=stories --vocab_size={vocab_size} --character_coverage=1.0 --model_type=unigram"
)
sp = spm.SentencePieceProcessor(model_file="./stories.model")

print("successfully trained sp")

PAD_TOKEN = sp.piece_to_id("<unk>")
batch_size = 16


def target_story_to_tensor(story):
    tokens = torch.tensor(
        sp.encode_as_ids(story) + [sp.piece_to_id("</s>")], dtype=torch.long
    )
    return tokens


def input_story_to_tensor(story):
    tokens = torch.tensor(
        [sp.piece_to_id("<s>")] + sp.encode_as_ids(story), dtype=torch.long
    )
    return tokens


class StoryDataset(Dataset):
    def __init__(self, stories):
        self.stories = stories

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        return input_story_to_tensor(story), target_story_to_tensor(story)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    targets = pad_sequence(targets, batch_first=True, padding_value=PAD_TOKEN)
    return inputs, targets


# Create dataset and dataloader
dataset = StoryDataset(text_data)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_data = StoryDataset(validation_data)
val_dataloader = DataLoader(
    val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

print(len(text_data))
print(len(validation_data))

# Hyperparameters
d_model = 64
dropout = 0.1  # 10% chance that any given neuron will be dropped out
n_heads = 8
n_layer = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


def create_mask(seq):
    seq_len = seq.size(1)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=seq.device), diagonal=1
    ).bool()
    return mask


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.blocks = nn.Sequential(*[Block(n_heads) for _ in range(n_layer)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input):
        # print('Im here')
        input = self.embedding(input)
        input = self.pos_encoder(input)
        blocks_output = self.blocks(input)

        logits = self.fc(blocks_output)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        head_size = d_model // n_heads
        self.multi_head_attention = MultiHeadAttention(n_heads, head_size)
        self.ffwd = nn.Sequential(
            nn.Linear(
                d_model, 4 * d_model
            ),  # expanding and contracting the model for it to learn more intricate patterns
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # actually doing residual connection here by attn1_output + input
        # print('im in block forward')
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        # print('Block shape', x.shape)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mask = create_mask(x).to(x.device)
        # print('im in multiheadattention')
        out = torch.cat(
            [h(x, x, x, mask) for h in self.heads], dim=-1
        )  # can parallelize it
        # print('multiheadattention out.shape', out.shape)
        out = self.dropout(self.proj(out))
        # here not printing
        # print('dropout out.shape', out.shape)
        return out


class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

    def forward(self, query, key, value, mask=None):
        # print('im in self attention')
        # print('head_size')
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        # print('q.shape', q.shape)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
        # print('scores.shape', scores.shape)
        # print('mask', mask)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        # print('attention_weights', attention_weights.shape)
        # print('v', v.shape)
        output = torch.matmul(attention_weights, v)
        # print('output.shape of selfattention', output.shape)
        return output


def plot_attention(attention, source_seq, target_seq):
    """
    Plots the attention weights.
    :param attention: Attention weights matrix.
    :param source_seq: Source sequence tokens.
    :param target_seq: Target sequence tokens.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        attention, cmap="viridis", xticklabels=source_seq, yticklabels=target_seq
    )
    plt.xlabel("Keys (Source)")
    plt.ylabel("Queries (Target)")
    plt.show()


import wandb
import time

num_epochs = 3
lr = 0.0001

name = "experiment-3"

# Define the model, optimizer, and loss
decoder = TransformerDecoder(vocab_size, d_model).to(device)
optimizer = Adam(decoder.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
num_params = sum(p.numel() for p in decoder.parameters())

# Training loop

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

run = wandb.init(
    project="story_generator",
    name=name,
    config={
        "optimizer": "Adam",
        "lr": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "n_heads": n_heads,
        "n_layer": n_layer,
        "d_model": d_model,
        "num_params": num_params,
    },
)

wandb.watch(decoder)


train_iteration = 0
val_iteration = 0
for epoch in range(num_epochs):
    decoder.train()
    total_loss = 0.0
    # start_time = time.time()
    for inputs, targets in dataloader:
        start_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = decoder(inputs)
        torch.cuda.empty_cache()
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_iteration += 1

        if train_iteration % 1000 == 0:
            avg_loss = total_loss / 1000
            end_time = time.time()
            iteration_duration = (end_time - start_time) * 1000
            wandb.log(
                {
                    "iteration": train_iteration,
                    "train_loss": avg_loss,
                    "time_taken": iteration_duration,
                }
            )
            total_loss = 0.0

    # end_time = time.time()  # <-- Record the end time
    # epoch_duration = end_time - start_time  # <-- Calculate epoch duration

    # avg_loss = total_loss / len(dataloader)
    # print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
    # wandb.log({"epoch": epoch, "train_loss": avg_loss, "time_taken": epoch_duration})

    # validation loop
    decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_dataloader:
            start_time = time.time()
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = decoder(val_inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            val_loss += loss.item()

            val_iteration += 1
            if val_iteration % 1000 == 0:
                avg_loss = val_loss / 1000
                end_time = time.time()
                iteration_duration = (end_time - start_time) * 1000
                wandb.log(
                    {
                        "iteration": val_iteration,
                        "val_loss": avg_loss,
                        "time_taken": iteration_duration,
                    }
                )
                val_loss = 0.0

    # avg_val_loss = val_loss / len(val_dataloader)
    # wandb.log({"epoch": epoch, "val_loss": avg_loss, "time_taken": epoch_duration})


# Save your model.
model_path = f"cuda-train-50000-{name}.pth"

wandb.log({"num_params": num_params})
torch.save(decoder.state_dict(), model_path)
artifact = wandb.Artifact("model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)
run.finish()


# torch.save(decoder.state_dict(), model_path)
# wandb.save(model_path)

print("vocab_size", vocab_size)


def generate_story(model, device, max_length=200):
    model.eval()
    with torch.no_grad():
        input_token = sp.piece_to_id("<s>")
        output_sequence = [
            input_token
        ]  # we'll always get the same name because we are using the same model and the same starter token

        for i in range(max_length):
            input_tensor = (
                torch.tensor([output_sequence]).long().to(device)
            )  # Move tensor to the correct device
            logit_output = model(input_tensor)

            softmax = nn.Softmax(dim=-1)
            softmax_output = softmax(logit_output)
            # Taking the token with the highest probability for prediction
            predicted_token = softmax_output[0, -1, :].argmax().item()

            # Break if we predict the end-of-string token
            if predicted_token == sp.piece_to_id("</s>"):
                break

            output_sequence.append(predicted_token)

        # Convert token IDs back to strings
        print(output_sequence[1:])
        generated_story = sp.decode_ids(output_sequence[1:])

    return generated_story


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generated_story = generate_story(
    decoder.to(device), device
)  # Make sure your model is on the correct device
print(generated_story)


model_path = "cuda-train-50000-epoch-3.pth"

# Save your model.
torch.save(decoder.state_dict(), model_path)
# Save as artifact for version control.
run = wandb.init(project="story-generator")
artifact = wandb.Artifact("model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)
run.finish()

run = wandb.init()

artifact = run.use_artifact("serena_chan/story-generator/model:v0", type="model")
artifact_dir = artifact.download()

model_path = os.path.join(artifact_dir, "cuda-train-50000-epoch-3.pth")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize your model architecture
decoder = TransformerDecoder(vocab_size, d_model).to(device)
# Load the state dictionary
decoder.load_state_dict(torch.load(model_path))

run.finish()

print("vocab_size", vocab_size)

temperature = 1.5


def temperature_sampling(logits):
    # Divide the logits by the temperature
    logits = logits / temperature
    # Create a distribution
    distribution = torch.nn.functional.softmax(logits, dim=-1)
    # Sample from the distribution
    choice = torch.multinomial(distribution, 1)
    token = choice.squeeze().item()
    return token


def generate_story(model, device, max_length=200):
    model.eval()
    with torch.no_grad():
        input_token = sp.piece_to_id("<s>")
        output_sequence = [
            input_token
        ]  # we'll always get the same name because we are using the same model and the same starter token

        for i in range(max_length):
            input_tensor = (
                torch.tensor([output_sequence]).long().to(device)
            )  # Move tensor to the correct device
            logit_output = model(input_tensor)

            predicted_token = temperature_sampling(logit_output[0, -1, :])

            # Break if we predict the end-of-string token
            if predicted_token == sp.piece_to_id("</s>"):
                break

            output_sequence.append(predicted_token)

        generated_story = sp.decode_ids(output_sequence[1:])

    return generated_story


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generated_story = generate_story(
    decoder, device
)  # Make sure your model is on the correct device
print(generated_story)
