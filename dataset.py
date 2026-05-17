import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy
from collections import Counter

class Multi30kDataset(Dataset):

    def __init__(self, split='train', min_freq=2, max_length=100,
        src_vocab=None,
        tgt_vocab=None,
        src_itos=None,
        tgt_itos=None,):
        """
        Multi30k German -> English dataset
        """

        super().__init__()

        self.split = split
        self.min_freq = min_freq
        self.max_length = max_length
        self.dataset = load_dataset(
            "bentrevett/multi30k",
            split=split
        )

        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")
        self.UNK_TOKEN = "<unk>"
        self.PAD_TOKEN = "<pad>"
        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"

        self.special_tokens = [
            self.UNK_TOKEN,
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
        ]

        if (src_vocab is None or tgt_vocab is None or src_itos is None or tgt_itos is None):
            self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

            self.src_itos = src_itos
            self.tgt_itos = tgt_itos

            self.src_pad_idx = self.src_vocab[self.PAD_TOKEN]
            self.tgt_pad_idx = self.tgt_vocab[self.PAD_TOKEN]

            self.src_sos_idx = self.src_vocab[self.SOS_TOKEN]
            self.tgt_sos_idx = self.tgt_vocab[self.SOS_TOKEN]

            self.src_eos_idx = self.src_vocab[self.EOS_TOKEN]
            self.tgt_eos_idx = self.tgt_vocab[self.EOS_TOKEN]

            self.src_unk_idx = self.src_vocab[self.UNK_TOKEN]
            self.tgt_unk_idx = self.tgt_vocab[self.UNK_TOKEN]
        self.process_data()

    def tokenize_de(self, text):
        return [token.text.lower() for token in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [token.text.lower() for token in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        """
        Build vocabularies from TRAIN split.
        """
        train_data = load_dataset("bentrevett/multi30k", split="train")
        de_counter = Counter()
        en_counter = Counter()

        for item in train_data:
            de_tokens = self.tokenize_de(item["de"])
            en_tokens = self.tokenize_en(item["en"])

            de_counter.update(de_tokens)
            en_counter.update(en_tokens)

        # stoi
        self.src_vocab = {}
        self.tgt_vocab = {}

        # itos
        self.src_itos = []
        self.tgt_itos = []

        for token in self.special_tokens:

            self.src_vocab[token] = len(self.src_vocab)
            self.src_itos.append(token)

            self.tgt_vocab[token] = len(self.tgt_vocab)
            self.tgt_itos.append(token)

        for token, freq in de_counter.items():

            if freq >= self.min_freq:
                if token not in self.src_vocab:
                    self.src_vocab[token] = len(self.src_vocab)
                    self.src_itos.append(token)

        for token, freq in en_counter.items():

            if freq >= self.min_freq:
                if token not in self.tgt_vocab:
                    self.tgt_vocab[token] = len(self.tgt_vocab)
                    self.tgt_itos.append(token)

        self.src_pad_idx = self.src_vocab[self.PAD_TOKEN]
        self.tgt_pad_idx = self.tgt_vocab[self.PAD_TOKEN]

        self.src_sos_idx = self.src_vocab[self.SOS_TOKEN]
        self.tgt_sos_idx = self.tgt_vocab[self.SOS_TOKEN]

        self.src_eos_idx = self.src_vocab[self.EOS_TOKEN]
        self.tgt_eos_idx = self.tgt_vocab[self.EOS_TOKEN]

        self.src_unk_idx = self.src_vocab[self.UNK_TOKEN]
        self.tgt_unk_idx = self.tgt_vocab[self.UNK_TOKEN]

    def numericalize_src(self, tokens):

        return [self.src_vocab.get(token, self.src_unk_idx) for token in tokens]

    def numericalize_tgt(self, tokens):
        return [self.tgt_vocab.get(token, self.tgt_unk_idx) for token in tokens]

    def process_data(self):
        """
        Process dataset.
        """
        self.examples = []

        for item in self.dataset:

            de_tokens = self.tokenize_de(item["de"])
            en_tokens = self.tokenize_en(item["en"])

            # Truncate
            de_tokens = de_tokens[:self.max_length - 2]
            en_tokens = en_tokens[:self.max_length - 2]

            # Add SOS/EOS
            de_tokens = ([self.SOS_TOKEN] + de_tokens + [self.EOS_TOKEN])

            en_tokens = ([self.SOS_TOKEN] + en_tokens + [self.EOS_TOKEN])

            src_ids = self.numericalize_src(de_tokens)
            tgt_ids = self.numericalize_tgt(en_tokens)

            self.examples.append((torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_idx=1):

    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_idx)

    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_idx)

    return src_batch, tgt_batch