from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pyarrow as pa

from config import get_weights_file_path, get_config

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import warnings

# Initialize the PyTorch distributed backend
# This is essential even for single machine, multiple GPU training
# dist.init_process_group(backend='nccl', init_method='tcp://localhost:FREE_PORT', world_size=1, rank=0)


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token we get from decoder
    # (input, mask)
    encoder_output = model.encode(source, source_mask)

    # Init the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        # no need for other mask because we do not have padding here
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token after the last we have given to the decoder
        prob = model.project(out[:, -1])
        # Select the token with the max prob (greedy search)
        _, next_word = torch.max(prob, dim=1)
        # Append to the decoder input for the next iter
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_state,
    writer=None,
    num_examples=2,
):
    model.eval()
    count = 0

    # source_texts = []
    # expected = []
    # predicted = []

    # Size of the control window (just use a default value)
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            # Print to the console
            print_msg("-" * console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break
    # If tensorboard enabled
    if writer:
        # TorchMetrics CharErrorRate, BLEU, WordErrorRate
        pass


def get_all_sentences(ds, lang):
    return ds[lang]


def get_or_build_tokenizer(config, ds, lang):
    # ~ config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = pd.read_csv(config["input"])

    le = LabelEncoder()
    le.fit(ds_raw["label"])

    ds_raw["le_label"] = le.transform(ds_raw["label"])
    ds_raw = Dataset(pa.Table.from_pandas(ds_raw))

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["tgt"])

    # Keep 90% training and 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["src"],
        config["tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["src"],
        config["tgt"],
        config["seq_len"],
    )

    # max_len_src = 0
    # max_len_tgt = 0

    # for item in ds_raw:
    #     src_ids = tokenizer_src.encode(item["translation"][config["src"]]).ids
    #     tgt_ids = tokenizer_tgt.encode(item["translation"][config["tgt"]]).ids
    #     max_len_src = max(max_len_src, len(src_ids))
    #     max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # print(f"Max length of source sentence: {max_len_src}")
    # print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, le


def get_model(config, vocab_src_len, vocab_tgt_len, num_classes):
    # In case of too big model decrease h or N
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        num_classes,
        config["d_model"],
    )
    return model


def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, le = get_ds(config)

    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        len(le.classes_),
    ).to(device)

    # model = DistributedDataParallel(model)

    # Tensorboard
    # writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    train_loss_list = []
    val_loss_list = []

    # First pretrain the whole model on a task of translating
    # corrupted sequences to original ones
    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epochs {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)  # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            #  hides [PAD] tokens
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, seq_len)
            #  hides subsequent tokens
            decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)  # (B, seq_len)

            # (B, seq_len, tgt_vocab_size) -> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )

            batch_iterator.set_postfix({f"loss:": f"{loss.item():6.3f}"})

            # Log the loss on tensorboard
            # writer.add_scalar("train_loss", loss.item(), global_step)
            # writer.flush()

            # Backprop
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            # run_validation(
            #     model,
            #     val_dataloader,
            #     tokenizer_src,
            #     tokenizer_tgt,
            #     config["seq_len"],
            #     device,
            #     lambda msg: batch_iterator.write(msg),
            #     global_step,
            #     writer,
            # )

            global_step += 1

        # Save model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02f}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )

    # Then fine-tune the classifier using the encoder embeddings to predict protein families
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze only the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    initial_epoch = 0
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=config["lr"], eps=1e-9
    )

    for epoch in range(config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epochs {epoch:02d}")
        train_loss = 0
        for batch in batch_iterator:
            model.train()
            enc_class_input = batch["enc_class_input"].to(device)  # (B, seq_len)
            enc_class_mask = batch["enc_class_mask"].to(device)  # (B, seq_len)
            enc_output = model.encode(
                enc_class_input, enc_class_mask
            )  # (B, seq_len, d_model)

            # Approach 1: Mean Pooling
            # pooled_encoder_output = torch.mean(enc_output, dim=1)
            # Approach 2: Using [CLS] or in our case [SOS] token - 0th index
            # The first token of every sequence is always a special classification token ([CLS]).
            # The final hidden state corresponding to this token is used as the aggregate sequence representation
            # for classification tasks.
            sos_encoder_output = enc_output[:, 0, :]

            classifier_output = model.classifier(
                sos_encoder_output
            )  # or sos_encoder_output
            target = batch["family"].to(device)  # (B, 1)
            # (B, seq_len, src_vocab_size) -> (B * seq_len, tgt_vocab_size)
            # should be
            # (B, d_model) -> (B, len(le.classes_))
            loss = loss_fn(
                classifier_output,
                target,
            )
            train_loss += loss.item()
            batch_iterator.set_postfix({f"cl_loss:": f"{loss.item():6.3f}"})
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(train_loss)
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                enc_class_input = batch["enc_class_input"].to(device)  # (B, seq_len)
                enc_class_mask = batch["enc_class_mask"].to(device)  # (B, seq_len)
                enc_output = model.encode(
                    enc_class_input, enc_class_mask
                )  # (B, seq_len, d_model)
                sos_encoder_output = enc_output[:, 0, :]
                classifier_output = model.classifier(sos_encoder_output)
                target = batch["family"].to(device)  # (B, seq_len)
                # (B, seq_len, src_vocab_size) -> (B * seq_len, tgt_vocab_size)
                loss = loss_fn(
                    classifier_output,
                    target,
                )
                val_loss += loss.item()
        val_loss = val_loss / len(val_dataloader)
        val_loss_list.append(val_loss)
    # Save model at the end of every epoch
    model_filename = get_weights_file_path(config, f"fin")
    torch.save(
        {
            "epoch": 222,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        model_filename,
    )
    # Plot loss
    fig, ax = plt.subplots()
    ax.plot(val_loss_list, label="Validation")
    ax.plot(train_loss_list, label="Training")
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.legend(loc="upper left")
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.savefig(f"loss.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
