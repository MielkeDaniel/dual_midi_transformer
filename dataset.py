import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import MIDI
from midi_tokenizer import MIDITokenizerV2

from torch.utils.data import DataLoader

EXTENSION = [".mid", ".midi"]

def file_ext(f_name):
    return os.path.splitext(f_name)[1].lower()

class MidiDataset(Dataset):
    def __init__(self, tokenizer, midi_list=None, max_events=128, min_events=20, aug=True, rand_start=True):

        self.tokenizer = tokenizer
        self.midi_list = get_midi_list("./content/909_dataset") if midi_list is None else midi_list
        random.shuffle(self.midi_list)
        self.min_events = min_events
        self.max_events = max_events
        self.aug = aug
        self.rand_start = rand_start

    def __len__(self):
        return len(self.midi_list)

    def get_midi_list(self):
        return self.midi_list

    def load_midi(self, index):
        path = self.midi_list[index]
        try:
            with open(path, 'rb') as f:
                datas = f.read()
            mid = MIDI.midi2score(datas)
            if max([0] + [len(track) for track in mid[1:]]) == 0:
                raise ValueError("empty track")
            mid = self.tokenizer.tokenize(mid)
            if self.aug:
                mid = self.tokenizer.augment(mid)

            if len(mid) < self.min_events:
                raise ValueError("too short")

        except Exception:
            mid = self.load_midi(random.randint(0, self.__len__() - 1))
        return mid

    def __getitem__(self, index):
        mid = self.load_midi(index)
        mid = np.asarray(mid, dtype=np.int16)

        if self.rand_start and mid.shape[0] > self.max_events:
            if random.random() < 0.15:  # 30% chance to start at 0
                start_idx = 0
            else:
                start_idx = random.randrange(0, max(1, mid.shape[0] - self.max_events))
        else:
            start_idx = 0
        mid = mid[start_idx: start_idx + self.max_events]
        mid = mid.astype(np.int64)
        mid = torch.from_numpy(mid)
        return mid

    # ensures that all midi files have the same length in a batch
    def collate_fn(self, batch):
        max_events = max([len(mid) for mid in batch])
        batch = [F.pad(mid, (0, 0, 0, max_events - mid.shape[0]), mode="constant", value=self.tokenizer.pad_id) for mid in batch]
        batch = torch.stack(batch)
        return batch

def get_midi_list(path):
    files = list()
    for (dirpath, _, filenames) in os.walk(path):
        files += [os.path.join(dirpath, file) for file in filenames]
    file_count = len(files)  
    print(f"Found {file_count} files")
    return files


def gen_mid(seq, name, tokenizer):
    # print seq len with name
    print(name, len(seq))
    seq = seq.cpu() 
    seq_np = seq.numpy()
    midi_score = tokenizer.detokenize(seq_np)
    midi_data = MIDI.score2midi(midi_score)

    with open("./training_samples/" + name, 'wb') as f:
        f.write(midi_data)