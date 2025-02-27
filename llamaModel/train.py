import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import matplotlib.pyplot as plt
import tqdm
import random
from dataset import MidiDataset, gen_mid
from midi_tokenizer import MIDITokenizerV2
from DualLlama import MIDIModel, MIDIModelConfig
import MIDI


class Args:
    def __init__(self):
        self.event_context_size = 128
        self.token_context_size = 8
        self.n_embd = 1024
        self.event_head = 8
        self.token_head = 4
        self.event_depth = 12
        self.token_depth = 3
        self.dropout = 0.1
        self.batch_size = 16
        self.grad_accum = 2
        self.lr = 5e-5
        self.weight_decay = 0.01
        self.warmup = 100
        self.max_step = 8000
        self.validate_every = 50
        self.generate_every = 300
        self.save_every = 500
        self.print_stats_every = 50
        self.checkpoint = './checkpoints/model_step_3797_loss_0.9662.pt'  # Path to checkpoint or None
        self.data_path = './content/909_dataset'
        self.save_dir = './checkpoints'
        self.sample_sequences = False
        self.example_batch = 2


def setup_training():
    # Configure PyTorch
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists('./generation_samples'):
        os.makedirs('./generation_samples')


def load_checkpoint(model, checkpoint_path):
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    start_step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', 0)

    print(f'Resuming from step {start_step} with loss {loss:.4f}')
    return model, start_step


def save_model(model, step, loss, path):
    checkpoint_path = os.path.join(path, f'model_step_{step}_loss_{loss:.4f}.pt')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f'Model saved to {checkpoint_path}')


def plot_training_progress(train_losses, train_accs, val_losses, val_accs, validate_every=None):
    """
    Plot the training and validation losses and accuracies.
    Training data is plotted against training steps.
    Validation data is plotted against validation rounds.
    """
    current_step = len(train_losses)

    plt.figure(figsize=(15, 6))

    # Plot losses
    ax1 = plt.subplot(1, 2, 1)

    # For training, use step indices
    train_steps = list(range(1, current_step + 1))

    # For validation, use validation round numbers
    val_indices = list(range(1, len(val_losses) + 1))

    # Plot training losses
    if current_step > 1000:
        # Sample for large datasets
        sparse_idx = list(range(0, current_step, 10))
        ax1.plot([train_steps[i] for i in sparse_idx],
                 [train_losses[i] for i in sparse_idx],
                 'b-', alpha=0.3, label='Train (sampled)')
        # Always show recent detail
        ax1.plot(train_steps[-100:], train_losses[-100:], 'b-', linewidth=1.5, label='Train (recent)')
    else:
        ax1.plot(train_steps, train_losses, 'b-', alpha=0.6, label='Train')

    # Create second axis for validation
    ax2 = ax1.twiny()
    line_val, = ax2.plot(val_indices, val_losses, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Validation Round')
    ax2.set_xlim(0.5, len(val_losses) + 0.5)

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss (Step {current_step})')

    # Add validation line to ax1's legend
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(line_val)
    labels.append('Validation')
    ax1.legend(handles, labels, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    ax3 = plt.subplot(1, 2, 2)

    # Plot training accuracies
    if current_step > 1000:
        ax3.plot([train_steps[i] for i in sparse_idx],
                 [train_accs[i] for i in sparse_idx],
                 'b-', alpha=0.3, label='Train (sampled)')
        ax3.plot(train_steps[-100:], train_accs[-100:], 'b-', linewidth=1.5, label='Train (recent)')
    else:
        ax3.plot(train_steps, train_accs, 'b-', alpha=0.6, label='Train')

    # Create second axis for validation
    ax4 = ax3.twiny()
    line_val_acc, = ax4.plot(val_indices, val_accs, 'r-o', linewidth=2, markersize=4)
    ax4.set_xlabel('Validation Round')
    ax4.set_xlim(0.5, len(val_losses) + 0.5)

    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Accuracy')
    ax3.set_title(f'Accuracy (Step {current_step})')

    # Add validation line to ax3's legend
    handles, labels = ax3.get_legend_handles_labels()
    handles.append(line_val_acc)
    labels.append('Validation')
    ax3.legend(handles, labels, loc='best')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save both a step-specific version and the latest version
    # plt.savefig(os.path.join(args.save_dir, f'training_progress_step_{current_step}.png'), dpi=150)
    plt.savefig(os.path.join(args.save_dir, 'latest_training_progress.png'), dpi=150)
    plt.close()


def generate_sample(model, tokenizer, step, val_dataset):
    model.eval()
    with torch.no_grad():
        samples = model.generate(max_len=128, batch_size=args.example_batch, temp=0.8)
        samples = [tokenizer.detokenize(sample) for sample in samples]

        # Save MIDI files
        for i, midi in enumerate(samples):
            with open(f'./generation_samples/sample_step_{step}_{i}.mid', 'wb') as f:
                f.write(MIDI.score2midi(midi))

        # generate sequence with prompt
        prompt = val_dataset.load_midi(random.randint(0, len(val_dataset) - 1))
        prompt = np.asarray(prompt, dtype=np.int16)
        prompt = prompt[:40].astype(np.int64)

        sample = model.generate(max_len=128, batch_size=args.example_batch, prompt=prompt, temp=0.8)
        sample = [tokenizer.detokenize(sample) for sample in sample]

        # Save MIDI files
        for i, midi in enumerate(sample):
            with open(f'./generation_samples/sample_step_{step}_{i}_prompt.mid', 'wb') as f:
                f.write(MIDI.score2midi(midi))


def train(args):
    # Initialize tokenizer
    print('Initializing tokenizer...')
    tokenizer = MIDITokenizerV2()
    tokenizer.set_optimise_midi(True)

    # Load dataset
    print('Creating MIDI dataset...')
    dataset = MidiDataset(tokenizer, max_events=args.event_context_size)

    # Split dataset
    validation_split = int(len(dataset) * 0.1)
    train_dataset_len = len(dataset) - validation_split
    train_midi_list = dataset.get_midi_list()[:train_dataset_len]
    val_midi_list = dataset.get_midi_list()[train_dataset_len:]

    train_dataset = MidiDataset(tokenizer, train_midi_list, aug=True, rand_start=True)
    val_dataset = MidiDataset(tokenizer, val_midi_list, aug=False, rand_start=False)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")

    # Initialize model
    # Create model config based on args
    model_config = MIDIModelConfig.get_config(
        tokenizer_ver="v2",
        optimise_midi=True,
        n_layer=args.event_depth,
        n_head=args.event_head,
        n_embd=args.n_embd,
        n_inner=args.n_embd * 4
    )
    model = MIDIModel(model_config)
    model.cuda()

    # Initialize step counter
    nsteps = 0

    # Load checkpoint if specified
    if args.checkpoint is not None:
        model, nsteps = load_checkpoint(model, args.checkpoint)

    # Optimizer setup
    optimizer = optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'norm'])],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'norm'])],
             'weight_decay': 0.0}
        ],
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(step / args.warmup, 1.0) * max(0.0, float(args.max_step - step) / float(
            max(1, args.max_step - args.warmup)))
    )

    scaler = GradScaler()

    # Training metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # nsteps is already set above

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    print('Starting training...')
    while nsteps < args.max_step:
        # Training loop
        model.train()
        for i, batch in enumerate(tqdm.tqdm(train_loader)):
            batch = batch.cuda()
            with torch.amp.autocast('cuda'):
                x = batch[:, :-1].contiguous()
                y = batch[:, 1:].contiguous()

                if args.sample_sequences:
                    rand_idx = [-1] + random.sample(list(range(y.shape[1] - 2)), min(127, (y.shape[1] - 2) // 2))
                    y = y[:, rand_idx]

                # Forward pass using DualLlama
                hidden = model.forward(x)
                hidden = hidden.reshape(-1, hidden.shape[-1])
                y = y.reshape(-1, y.shape[-1])
                x = y[:, :-1]
                logits = model.forward_token(hidden, x)

                # sample from training
                if i % 1000 == 0:
                    targets = y
                    predictions = torch.argmax(logits, dim=-1)
                    gen_mid(targets, f"targets_{nsteps}_{i}.mid", tokenizer)
                    gen_mid(predictions, f"predictions_{nsteps}_{i}.mid", tokenizer)

                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, tokenizer.vocab_size),
                    y.view(-1),
                    ignore_index=tokenizer.pad_id
                )
                loss = loss / args.grad_accum

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = (y != tokenizer.pad_id)
                acc = (predictions[mask] == y[mask]).float().mean()

            scaler.scale(loss).backward()

            if ((i + 1) % args.grad_accum == 0) or (i + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss.item() * args.grad_accum)
            train_accs.append(acc.item())

            nsteps += 1

            # Print stats
            if i % args.print_stats_every == 0:
                print(
                    f'Step {nsteps} - Loss: {loss.item() * args.grad_accum:.4f}, Acc: {acc.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

            # Validation
            if i % args.validate_every == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    for val_batch in val_loader:
                        val_batch = val_batch.cuda()
                        with torch.cuda.amp.autocast():
                            x = val_batch[:, :-1].contiguous()
                            y = val_batch[:, 1:].contiguous()
                            hidden = model.forward(x)
                            hidden = hidden.reshape(-1, hidden.shape[-1])
                            y = y.reshape(-1, y.shape[-1])
                            x = y[:, :-1]
                            logits = model.forward_token(hidden, x)

                            loss = torch.nn.functional.cross_entropy(
                                logits.view(-1, tokenizer.vocab_size),
                                y.view(-1),
                                ignore_index=tokenizer.pad_id
                            )

                            predictions = torch.argmax(logits, dim=-1)
                            mask = (y != tokenizer.pad_id)
                            batch_acc = (predictions[mask] == y[mask]).float().mean()

                            val_loss += loss.item()
                            val_acc += batch_acc.item()

                    val_loss /= len(val_loader)
                    val_acc /= len(val_loader)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                    print(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
                    plot_training_progress(train_losses, train_accs, val_losses, val_accs, args.validate_every)

                model.train()

            # Generate sample
            if i % args.generate_every == 0:
                generate_sample(model, tokenizer, nsteps, val_dataset)

            # Save checkpoint
            if i % args.save_every == 0:
                save_model(model, nsteps, train_losses[-1], args.save_dir)

            if nsteps >= args.max_step:
                break

        if nsteps >= args.max_step:
            break

    # Save final model
    if train_losses:
        save_model(model, nsteps, train_losses[-1], args.save_dir)
    else:
        save_model(model, nsteps, 0.0, args.save_dir)


if __name__ == "__main__":
    args = Args()
    setup_training()
    train(args)