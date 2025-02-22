import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import MIDI
from midi_tokenizer import MIDITokenizerV2

from torch.utils.data import  DataLoader

EXTENSION = [".mid", ".midi"]

def file_ext(f_name):
    return os.path.splitext(f_name)[1].lower()

class MidiDataset(Dataset):
    def __init__(self, tokenizer, midi_list=None, max_len=96, aug=True, rand_start=True):

        self.tokenizer = tokenizer
        self.midi_list = get_midi_list("./dataset") if midi_list is None else midi_list
        #random.shuffle(self.midi_list)

        self.max_len = max_len
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
        except Exception:
            mid = self.load_midi(random.randint(0, self.__len__() - 1))
        return mid

    def __getitem__(self, index):
        mid = self.load_midi(index)
        mid = np.asarray(mid, dtype=np.int16)
        #if mid.shape[0] < self.max_len:
         #   mid = np.pad(mid, ((0, self.max_len - mid.shape[0]), (0, 0)),
          #              mode="constant", constant_values=self.tokenizer.pad_id)
        if self.rand_start:
            start_idx = random.randrange(0, max(1, mid.shape[0] - self.max_len))
            start_idx = random.choice([0, start_idx])
        else:
            max_start = max(1, mid.shape[0] - self.max_len)
            start_idx = (index * (max_start // 8)) % max_start
        mid = mid[start_idx: start_idx + self.max_len]
        mid = mid.astype(np.int64)
        mid = torch.from_numpy(mid)
        return mid

    # ensures that all midi files have the same length in a batch
    def collate_fn(self, batch):
        max_len = max([len(mid) for mid in batch])
        batch = [F.pad(mid, (0, 0, 0, max_len - mid.shape[0]), mode="constant", value=self.tokenizer.pad_id) for mid in batch]
        batch = torch.stack(batch)
        return batch

def get_midi_list(path):
    all_files = {
        os.path.join(root, f_name)
        for root, _dirs, files in os.walk(path)
        for f_name in files
    }
    all_midis = sorted(
        f_name for f_name in all_files if file_ext(f_name) in EXTENSION
    )
    full_dataset_len = len(all_midis)
    print(f"Found {full_dataset_len} MIDI files")
    return all_midis

# test code
if __name__ == '__main__':
    dataset = MidiDataset(MIDITokenizerV2())
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn, persistent_workers=True, num_workers=4, pin_memory=True)
    batch = next(iter(dataloader))
    print(batch.shape)
