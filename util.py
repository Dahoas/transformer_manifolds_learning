import json
import os
from typing import List, Union

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from model import GPT, GPTConfig
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class DNATokenizer:
    nuc_to_ind = {
                        "A": 0,
                        "G": 1, 
                        "T": 2,
                        "C": 3,
                        "<|endoftext|>": 4,
                        "N": 5,
                    }
    ind_to_nuc = {
                        0: "A",
                        1: "G",
                        2: "T",
                        3: "C",
                        4: "<|endoftext|>",
                        5: "N",
                    }
    eos_ind = 5
    eos_token_id = eos_ind

    def __init__(self):
        pass

    def __call__(self, seq: str, add_eos=False,):
        try:
            inds = [self.nuc_to_ind[nuc] for nuc in seq]
            inds += [self.eos_ind] if add_eos else []
            return inds
        except KeyError:
            return []

    def tokenize(row):
        c_name, start, end = row["chr_name"], row["start"],  row["end"]
        chromosome = fasta[c_name]
        seq = str(chromosome[start:end])
        seq_toks = tokenizer(seq, add_eos=True)
        return {"ids": seq_toks}


def jsonl_to_dict(samples: list[dict]):
    return {k: [sample[k] for sample in samples] for k in samples[0]}


def load_data(path: str, mode: str, split: str=None, tokenizer=None, streaming=False,):
    if mode == "json":
        with open(path, "r") as f:
            data = jsonl_to_dict(json.load(f))
            data = {"train": data}
    elif mode == "jsonl":
        with open(path, "r") as f:
            data = f.readlines()
            data = jsonl_to_dict([json.loads(sample) for sample in data])
    elif mode == "hf":
        data = load_dataset(path, split=split, streaming=streaming)
    elif mode == "token":
        print("Loading dataset...")
        def recover_docs(data, eos_token_id):
            print("Searching for eos inds...")
            eos_inds = [-1] + list((data == eos_token_id).nonzero()[0])
            docs = []
            print("Iterating...")
            for i in tqdm(range(len(eos_inds)-1), total=len(eos_inds)):
                start, end = eos_inds[i], eos_inds[i+1]
                doc = data[start+1:end]
                docs.append(doc)
            return docs
        eos_token_id = tokenizer.eos_token_id
        train_path = os.path.join(path, f"train.bin")
        if split is None or split == "train":
            train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
            train_data = recover_docs(train_data, eos_token_id)
        validation_path = os.path.join(path, "val.bin")
        if split is None or split == "validation":
            val_data = np.memmap(validation_path, dtype=np.uint16, mode="r")
            val_data = recover_docs(val_data, eos_token_id)
        if split is None:
            data = dict(train=dict(text=train_data), validation=dict(text=val_data))
        elif split == "train":
            data = dict(text=train_data)
        elif split == "validation":
            data = dict(text=val_data)
    return data


class Tokenizer:
    def __init__(self, tokenizer_path, tokenizer_type, dataset_mode):
        self.tokenizer_type = tokenizer_type
        self.dataset_mode = dataset_mode
        if self.tokenizer_type == "oai":
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.eos_token_id = self.tokenizer.eot_token
        elif self.tokenizer_type == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.eos_token_id = self.tokenizer.eos_token_id
        elif self.tokenizer_type == "dna":
            self.tokenizer = DNATokenizer()
            self.eos_token_id = self.tokenizer.eos_token_id
                

    def _call_hf_tokenizer(self,
                           batch,
                           max_length,):
        return self.tokenizer(batch, truncation=True, max_length=max_length, return_tensors="pt", padding="max_length")

    def _call_oai_tokenizer(self,
                            batch: List[Union[str, List[int]]],
                            max_length: int,):
        # Encode and truncate output
        if self.dataset_mode != "token":
            input_ids = [self.tokenizer.encode_ordinary(sample)[:max_length] for sample in batch]
        else:
            input_ids = [list(sample[:max_length]) for sample in batch]
        attention_mask = [list(np.ones(len(ids))) for ids in input_ids]
        # Pad to max length
        input_ids = [encoding + (max(0, max_length - len(encoding)) * [self.tokenizer.eot_token]) for encoding in input_ids]
        attention_mask = [mask + (max(0, max_length - len(mask)) * [0]) for mask in attention_mask]
        # Convert output into pt tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return dict(input_ids=input_ids, attention_mask=attention_mask)

    def _call_dna_tokenizer(self,
                             batch,
                             max_length,):
        # Filter out samples with no tokens
        batch = [list(tokens[:max_length]) + max(0, max_length - len(tokens)) * [self.eos_token_id] for tokens in batch if len(tokens) > 0]
        input_ids = torch.tensor(batch, dtype=torch.long)
        return dict(input_ids=input_ids, attention_mask=torch.empty((len(input_ids), max_length)))

    def __call__(self, 
                 batch,
                 max_length,
                 **kwargs,):
        if self.tokenizer_type == "hf":
            return self._call_hf_tokenizer(batch, max_length)
        elif self.tokenizer_type == "oai":
            return self._call_oai_tokenizer(batch, max_length)
        if self.tokenizer_type == "dna":
            return self._call_dna_tokenizer(batch, max_length)


def load_model_tokenizer(model_path, model_mode, tokenizer_type, tokenizer_path, dataset_mode,):
    if model_mode == "hf":
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = model.half()
        tokenizer = Tokenizer(tokenizer_path=tokenizer_path, tokenizer_type=tokenizer_type, dataset_mode=dataset_mode)
        config = AutoConfig.from_pretrained(model_path)
        return model, tokenizer, config.num_hidden_layers
    elif model_mode == "oai":
        if ".pt" not in model_path:
            ckpt_path = os.path.join(model_path, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            checkpoint_model_args = checkpoint['model_args']
            state_dict = checkpoint['model']
        else:
            state_dict = torch.load(model_path)
            dir_path = os.path.dirname(model_path)
            ckpt_path = os.path.join(dir_path, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        # create the model
        gptconf = GPTConfig(**checkpoint_model_args)
        model = GPT(gptconf)
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        tokenizer = Tokenizer(tokenizer_path=tokenizer_path, tokenizer_type=tokenizer_type, dataset_mode=dataset_mode)
        return model, tokenizer, gptconf.n_layer