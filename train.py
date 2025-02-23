import os
import random
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import  DataLoader
from dataset import MidiDataset

import MIDI
from dual_transformer import DualTransformer
from midi_tokenizer import MIDITokenizerV2

# Dataset parameters
data_path = "./dataset"
validation_split_percentage = 0.1 # Fraction of dataset to use for validation

# Training parameters
random_seed = 32
learning_rate = 2e-4
weight_decay = 0.01
warmup_steps = 100
max_training_steps = 50000
gradient_clip = 1.0
sample_sequences = False  # Whether to sample MIDI sequences to reduce VRAM
generation_interval = 1  # Set to 0 to disable example generation
training_batch_size = 8
validation_batch_size = 8
example_generation_batch_size = 2
training_workers = 4
validation_workers = 4
gradient_accumulation = 2

# Hardware configuration
accelerator = "gpu"  # Options: "cpu", "gpu", "tpu", "ipu", "hpu", "auto"
precision = "32-true"  # Options: "16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true"
num_devices = -1
num_nodes = 1
disable_cudnn_benchmark = False
log_frequency = 2  # Log training loss every n steps
validation_frequency = 0  # Validate every n steps (0 = once per epoch)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class TrainMIDIModel(DualTransformer, pl.LightningModule):
    def __init__(self, tokenizer,
                 lr=2e-4, weight_decay=0.01, warmup=1e3, max_step=1e6,
                 sample_seq=False, gen_example_interval=1, example_batch=8,
                 event_context_size=128, token_context_size=8,
                 n_embd=512, event_head=8, token_head=2,
                 event_depth=6, token_depth=2, dropout=0.1):
        super().__init__(
            vocab_size=tokenizer.vocab_size,
            event_context_size=event_context_size,
            token_context_size=token_context_size,
            n_embd=n_embd,
            event_head=event_head,
            token_head=token_head,
            event_depth=event_depth,
            token_depth=token_depth,
            dropout=dropout
        )
        self.tokenizer = tokenizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.max_step = max_step
        self.sample_seq = sample_seq
        self.gen_example_interval = gen_example_interval
        self.example_batch = example_batch
        self.last_save_step = 0
        self.gen_example_count = 0

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'norm']  # no decay for bias and Norm
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay},
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup,
            num_training_steps=self.max_step,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def compute_accuracy(self, logits, labels):
        out = torch.argmax(logits, dim=-1)
        out = out.flatten()
        labels = labels.flatten()

        mask = (labels != self.tokenizer.pad_id)
        out = out[mask]
        labels = labels[mask]

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)
        acc = num_right / len(labels)

        return acc

    def training_step(self, batch, batch_idx):
        x = batch[:, :-1].contiguous()  # (batch_size, midi_sequence_length, token_sequence_length)
        y = batch[:, 1:].contiguous()
        hidden = self.forward(x)
        if self.sample_seq:  # to reduce vram
            rand_idx = [-1] + random.sample(list(range(y.shape[1] - 2)), min(127, (y.shape[1] - 2) // 2))
            hidden = hidden[:, rand_idx]
            y = y[:, rand_idx]
        hidden = hidden.reshape(-1, hidden.shape[-1])
        y = y.reshape(-1, y.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
        x = y[:, :-1]
        logits = self.forward_token(hidden, x)
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            y.view(-1),
            reduction="mean",
            ignore_index=self.tokenizer.pad_id
        )
        self.log("train/loss", loss)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, :-1].contiguous()  # (batch_size, midi_sequence_length, token_sequence_length)
        y = batch[:, 1:].contiguous()
        hidden = self.forward(x)
        hidden = hidden.reshape(-1, hidden.shape[-1])
        y = y.reshape(-1, y.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
        x = y[:, :-1]
        logits = self.forward_token(hidden, x)
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            y.view(-1),
            reduction="mean",
            ignore_index=self.tokenizer.pad_id
        )
        acc = self.compute_accuracy(logits, y)
        self.log_dict({"val/loss": loss, "val/acc": acc}, sync_dist=True)
        return loss

    @rank_zero_only
    def gen_example(self, save_dir):
        base_dir = f"{save_dir}/sample/{self.global_step}"
        if not os.path.exists(base_dir):
            Path(base_dir).mkdir(parents=True)
        midis = self.generate(batch_size=self.example_batch, tokenizer=self.tokenizer)
        midis = [self.tokenizer.detokenize(midi) for midi in midis]
        imgs = [self.tokenizer.midi2img(midi) for midi in midis]
        for i, (img, midi) in enumerate(zip(imgs, midis)):
            img.save(f"{base_dir}/0_{i}.png")
            with open(f"{base_dir}/0_{i}.mid", 'wb') as f:
                f.write(MIDI.score2midi(midi))
        prompt = val_dataset.load_midi(random.randint(0, len(val_dataset) - 1))
        prompt = np.asarray(prompt, dtype=np.int16)
        ori = prompt[:512]
        img = self.tokenizer.midi2img(self.tokenizer.detokenize(ori))
        img.save(f"{base_dir}/1_ori.png")
        prompt = prompt[:256].astype(np.int64)
        midis = self.generate(prompt, batch_size=self.example_batch, tokenizer=self.tokenizer)
        midis = [self.tokenizer.detokenize(midi) for midi in midis]
        imgs = [self.tokenizer.midi2img(midi) for midi in midis]
        for i, (img, midi) in enumerate(zip(imgs, midis)):
            img.save(f"{base_dir}/1_{i}.png")
            with open(f"{base_dir}/1_{i}.mid", 'wb') as f:
                f.write(MIDI.score2midi(midi))

    def on_save_checkpoint(self, checkpoint):
        if self.global_step == self.last_save_step:
            return
        self.last_save_step = self.global_step
        trainer = self.trainer
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            save_dir = os.path.join(save_dir, str(name), version)
        else:
            save_dir = trainer.default_root_dir
        self.gen_example_count += 1
        if self.gen_example_interval>0 and self.gen_example_count % self.gen_example_interval == 0:
            try:
                self.gen_example(save_dir)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs("lightning_logs", exist_ok=True)
    os.makedirs("sample", exist_ok=True)

    # Set random seed
    pl.seed_everything(random_seed)

    print("---load dataset---")
    tokenizer = MIDITokenizerV2()

    # Initialize datasets
    midiDataset = MidiDataset(tokenizer)
    validation_split = int(len(midiDataset) * validation_split_percentage)
    train_dataset_len = len(midiDataset) - validation_split
    train_midi_list = midiDataset.get_midi_list()[:train_dataset_len]
    val_midi_list = midiDataset.get_midi_list()[train_dataset_len:]

    train_dataset = MidiDataset(tokenizer, train_midi_list, aug=True, rand_start=True)
    val_dataset = MidiDataset(tokenizer, val_midi_list, aug=False, rand_start=False)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        persistent_workers=True,
        num_workers=training_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        persistent_workers=True,
        num_workers=validation_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    # Enable CUDA optimizations
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    # Initialize model
    model = TrainMIDIModel(
        tokenizer=tokenizer,
        lr=learning_rate,
        weight_decay=weight_decay,
        warmup=warmup_steps,
        max_step=max_training_steps,
        sample_seq=sample_sequences,
        gen_example_interval=generation_interval,
        example_batch=example_generation_batch_size,
        n_embd=512,
        event_head=8,
        token_head=2,
        event_depth=6,
        token_depth=2,
        dropout=0.1
    )

    # Setup training callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        filename="epoch={epoch},loss={val/loss:.4f}",
    )

    # Initialize trainer
    trainer = Trainer(
        precision=precision,
        accumulate_grad_batches=gradient_accumulation,
        gradient_clip_val=gradient_clip,
        accelerator=accelerator,
        devices=num_devices,
        num_nodes=num_nodes,
        max_steps=max_training_steps,
        benchmark=not disable_cudnn_benchmark,
        val_check_interval=validation_frequency or None,
        log_every_n_steps=log_frequency,
        strategy="auto",
        callbacks=[checkpoint_callback],
    )

    # Start training
    print("---start train---")
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=None
    )