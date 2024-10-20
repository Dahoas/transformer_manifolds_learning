import argparse
import datetime
import json
import os
import pathlib
import re
from collections import defaultdict
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
import torch
from skdim.id import MLE
from tqdm import tqdm
from util import load_data, load_model_tokenizer

MAX_LENGTH = 512
batch_size = 16
log_size = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EMBEDDINGS = False
id_estimation_batch_size = 3600


def get_embeddings(texts: List[str],
                   tokenizer,
                   model,
                   layers=[-1],
                   keep_eos_embeds=False,
                   compute_loss=False,
                   take_random_subseq=False,) -> dict[int, list[torch.tensor]]:
    embedding_list = {layer: [] for layer in layers}
    log_list = {layer: [] for layer in layers}
    num_batches = (len(texts) + batch_size - 1) // batch_size
    batches = [texts[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    losses = []
    i = 0
    while len(batches) > 0:
        print(f"Batch {i} of {num_batches}")
        batch = batches.pop(0)
        if take_random_subseq:
            inputs = tokenizer(batch, max_length=100_000)["input_ids"]
            subseq_inputs = {"input_ids": [], "attention_mask": []}
            for inp in inputs:
                ind = np.random.randint(low=0, high=max(0, len(inp)-MAX_LENGTH))
                inp = inp[ind:ind+MAX_LENGTH]
                inp = torch.cat([inp, torch.zeros(MAX_LENGTH-len(inp), dtype=torch.long)+tokenizer.eos_token_id])
                subseq_inputs["input_ids"].append(inp)
                subseq_inputs["attention_mask"].append(torch.ones(MAX_LENGTH, dtype=torch.long))
            subseq_inputs["input_ids"] = torch.stack(subseq_inputs["input_ids"])
            subseq_inputs["attention_mask"] = torch.stack(subseq_inputs["attention_mask"])
            inputs = subseq_inputs
        else:
            inputs = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, return_tensors="pt", padding="max_length")
        # TODO(alex): I am only taking the prefixes of documents. Should really be randomly sampling position in each document
        if len(inputs["input_ids"]) == 0:
            i += 1
            del batch
            continue
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        with torch.no_grad():
            if compute_loss:
                labels = inputs["input_ids"] * (inputs["input_ids"] != tokenizer.eos_token_id) + \
                            (torch.zeros_like(inputs["input_ids"], dtype=torch.long)-100) * (inputs["input_ids"] == tokenizer.eos_token_id) #[:, 1:]
                input_ids = inputs["input_ids"]  #[:, :-1]
                attention_mask = inputs["attention_mask"]#[:, :-1]
                output = model(input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               labels=labels)
                losses.append(output.loss)
            else:
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states
                for layer in layers:
                    embeddings = hidden_states[layer].cpu()
                    log_list[layer].append(embeddings)
                    # Remove EOS tokens
                    mask = inputs["input_ids"] != tokenizer.eos_token_id
                    for m, embedding in zip(mask, embeddings):
                        if not keep_eos_embeds:
                            embedding = embedding[:m.sum()]
                        embedding_list[layer].append(embedding)
                    if len(log_list[layer]) >= log_size and LOG_EMBEDDINGS:
                        print("Logging embeddings...")
                        log_embeddings = torch.cat(log_list[layer], dim=0)
                        torch.save(log_embeddings, os.path.join(log_folder, f"embeddings_layer_{layer}_{i}.pt"))
                        log_list[layer] = []
        i += 1
        del batch
    if compute_loss:
        return losses
    else:
        for layer in layers:
            if len(log_list[layer]) > 0 and LOG_EMBEDDINGS:
                print("Logging embeddings...")
                log_embeddings = torch.cat(log_list[layer], dim=0)
                torch.save(log_embeddings, os.path.join(log_folder, f"embeddings_layer_{layer}_{i+1}.pt"))
        return embedding_list

def load_embeddings(path):
    embeddings = defaultdict(list)
    if ".csv" in path:
        df = pd.read_csv(path)
        embeddings[-1] = [torch.tensor(eval(embedding), dtype=torch.float32).reshape((1, -1)) for embedding in df["embedding"]]
        return embeddings
    else:
        files = pathlib.Path(path).glob("*.pt")
        for f in files:
            print(f"Loading embeddings from {f}...")
            embds = torch.load(f)
            layer = int(re.findall(r"layer_(\d+)", str(f))[0])
            embeddings[layer] += list(embds)
            print(f"Loaded tensor of size {embds.shape} from {f}...")
        return embeddings

def get_stats(nums: List[float]):
    print(nums)
    filtered_nums = [num for num in nums if not np.isnan(num) and num != np.inf]
    l = len(nums)
    num_nan = l - len(filtered_nums)
    nums = filtered_nums
    return dict(mean=np.mean(nums),
            std=np.std(nums),
            min=np.min(nums),
            max=np.max(nums),
            len=l,
            num_nan=num_nan,)

class MLEMode(Enum):
    MEAN = 1
    ALL = 2
    STACK_CONTEXT = 3
    @classmethod
    def from_string(cls, name):
        name_to_val = {val.name: val for val in cls}
        if name_to_val.get(name.upper(), None):
            return name_to_val[name.upper()]
        else:
            raise ValueError(f"Unknown name: {name}!!!")


def get_mle(samples: List[torch.tensor], K=5, mode="mean", metric="minkoswki"):
    #TODO(dahoas): not entirely sure this is the right way of evaluating intrinsic dim
    mode = MLEMode.from_string(mode)
    mles = []
    solver = MLE(metric=metric)
    print("Dnoise: ", solver.dnoise)
    print("Neighborhood base: ", solver.neighborhood_based)
    extrinsic_dim = samples[0].shape[-1]
    if mode == MLEMode.MEAN:
        for sample in tqdm(samples):
            int_dim = solver.fit_transform(sample.numpy(), n_neighbors=K,)
            mles.append(int_dim)
        print(sample.shape)
        int_dim = np.mean(mles)
        stats = get_stats(mles)
        stats["extrinsic_dim"] = extrinsic_dim
        return int_dim, mles, stats
    elif mode == MLEMode.ALL:
        samples = torch.cat(samples, dim=0).numpy()
        print(samples.shape)
        num_batches = (len(samples) + id_estimation_batch_size - 1) // id_estimation_batch_size
        batches = [samples[id_estimation_batch_size*i:id_estimation_batch_size*(i+1)] for i in range(num_batches)]
        for i, batch in enumerate(batches):
            print(f"Batch num: {i} of {num_batches}", f"Batch shape: {batch.shape}")
            int_dim = solver.fit_transform(batch, n_neighbors=K,)
            mles.append(int_dim)
        int_dim = np.mean(mles)
        stats = get_stats(mles)
        stats["extrinsic_dim"] = extrinsic_dim
        return int_dim, mles, stats
    elif mode == MLEMode.STACK_CONTEXT:
        samples = torch.cat([sample.reshape((1,-1)) for sample in samples], dim=0)
        print(samples.shape)
        num_batches = (len(samples) + id_estimation_batch_size - 1) // id_estimation_batch_size
        batches = [samples[id_estimation_batch_size*i:id_estimation_batch_size*(i+1)] for i in range(num_batches)]
        for batch in batches:
            print(batch.shape)
            int_dim = solver.fit_transform(batch, n_neighbors=K,)
            mles.append(int_dim)
        int_dim = np.mean(mles)
        stats = get_stats(mles)
        stats["extrinsic_dim"] = extrinsic_dim
        return int_dim, mles, stats

def preprocessing(texts):
    """
    Applies (optional)preprocessing to improve embedding quality.
    """
    def preproc(text):
        return text.replace("\n", " ")
    new_texts = [preproc(text) for text in texts]
    return new_texts 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="roneneldan/TinyStories-33M")
    parser.add_argument("--tokenizer_type", default="hf", choices=["hf", "oai", "dna"])
    parser.add_argument("--tokenizer_path", default=None,)
    parser.add_argument("--model_mode", default="hf", choices=["hf", "oai"])
    
    parser.add_argument("--layers", default=[-1], nargs="+", type=int, help="Layers to measure ID of")
    
    parser.add_argument("--dataset_path", default="roneneldan/TinyStories")
    parser.add_argument("--dataset_mode", default="hf", choices=["hf", "token", "jsonl"])
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--data_key", default="text", type=str, help="Column name in dataset containing samples")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset")
    
    parser.add_argument("--dataset_lower", default=0, type=int, help="Lower index into documents")
    parser.add_argument("--dataset_upper", default=128, type=int, help="Upper index into documents")
    parser.add_argument("--num_dataset_subsample", default=None, type=int, help="Number of documents to randomly subsample before embedding")
    parser.add_argument("--max_embeddings", default=1_000_000, type=int, help="Limit number of token embeddings using to estimate ID.")
    parser.add_argument("--shuffle_embeddings", action="store_true", help="If true shuffles embeddings.")
    parser.add_argument("--max_embeddings_per_sample", default=None, type=int, help="Limit number of embeddings per sample")
    parser.add_argument("--shuffle_embeddings_per_sample", action="store_true", help="Shuffle embeddings for a single document. Should be used with max_embeddings per sample.")
    parser.add_argument("--context_len", default=1024, type=int, help="Max context length to embed")
    parser.add_argument("--take_random_subseq", action="store_true", help="If true randomly takes subseq of document (instead of prefix) of context_len")
    
    parser.add_argument("--keep_eos_embeds", action="store_true", help="Keep EOS tokens in ID estimation")
    parser.add_argument("--apply_preprocessing", action="store_true", help="Apply simple pre-processing to text")
    parser.add_argument("--num_token_groups", default=None, type=int, help="Groups embeddings based on position")

    parser.add_argument("--mle_mode", default="all", choices=["mean", "all", "stack_context"], help="Mode to calculate ID")
    parser.add_argument("--metric", default="cosine", choices=["minkowski", "cosine"], help="Distance metric to use.")
    parser.add_argument("--K", default=20, type=int, help="Number of neighbors to use")
    
    parser.add_argument("--compute_loss", action="store_true", help="If true will compute loss on dataset instead of id")
    parser.add_argument("--embedding_path", default=None, type=str, help="Path to pre-computed embeddings if available.")
    parser.add_argument("--batch_size", default=16, type=int) 
    parser.add_argument("--log_folder", default="logs")
    args = parser.parse_args()

    # Make logging folder
    log_folder = pathlib.Path(os.path.join(args.log_folder, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
    log_folder.mkdir(exist_ok=True, parents=True)
    print("Logging in: ", log_folder)
    with open(os.path.join(log_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model, tokenizer
    args.tokenizer_path = args.model_path if args.tokenizer_path is None else args.tokenizer_path
    model, tokenizer, num_hidden_layers = load_model_tokenizer(args.model_path, 
                                                               args.model_mode, 
                                                               args.tokenizer_type, 
                                                               args.tokenizer_path, 
                                                               args.dataset_mode)
    model = model.to(device)

    # Load data
    MAX_LENGTH = args.context_len
    batch_size = args.batch_size
    texts = load_data(args.dataset_path, 
                      mode=args.dataset_mode, 
                      split=args.split, 
                      tokenizer=tokenizer, 
                      streaming=args.streaming)
    if args.streaming:
        data = iter(texts)
        texts = []
        for i in range(args.dataset_upper):
            sample = next(data)[args.data_key]
            if i >= args.dataset_lower:
                texts.append(sample)
    else:
        texts = texts[args.data_key][args.dataset_lower:args.dataset_upper]
    if args.num_dataset_subsample:
        inds = np.random.choice(np.arange(len(texts)), size=(args.num_dataset_subsample,), replace=False)
        texts = [texts[ind] for ind in inds]
    if args.apply_preprocessing:
        texts = preprocessing(texts)

    if args.layers[-1] > num_hidden_layers:
        args.layers = [i for i in range(num_hidden_layers)]

    # Generate or load embeddings
    if not args.embedding_path:
        embeddings = get_embeddings(texts, 
                                    tokenizer, 
                                    model, 
                                    layers=args.layers, 
                                    keep_eos_embeds=args.keep_eos_embeds,
                                    compute_loss=args.compute_loss,
                                    take_random_subseq=args.take_random_subseq,)
        if args.compute_loss:
            loss = torch.stack(embeddings, dim=0).mean()
            print("Loss: ", loss.item())
            exit()
    else:
        embeddings = load_embeddings(args.embedding_path)

    del model
    del tokenizer
    del texts

    # Compute IDs
    all_stats = dict()
    for layer, layer_embeddings in embeddings.items():
        if args.num_token_groups:
            all_stats[layer] = dict()
            token_groups = [[] for _ in range(args.num_token_groups)]
            for sample in layer_embeddings:
                for i in range(args.num_token_groups):
                    token_lower = i * (MAX_LENGTH // args.num_token_groups)
                    token_upper = (i+1) * (MAX_LENGTH // args.num_token_groups)
                    token_groups[i].append(sample[token_lower:token_upper])
            for i, token_group in enumerate(token_groups):
                token_lower = i * (MAX_LENGTH // args.num_token_groups)
                token_upper = (i+1) * (MAX_LENGTH // args.num_token_groups)
                token_group = torch.cat(token_group, dim=0)
                if args.shuffle_embeddings:
                    token_group = token_group[torch.randperm(token_group.shape[0])]
                token_group = [token_group]
                int_dim, mles, stats = get_mle(token_group, mode=args.mle_mode, K=args.K, metric=args.metric)
                print("Layer: ", layer, "Token lower: ", token_lower, "Token upper: ", token_upper)
                print("Intrinsic dim: ", int_dim)
                stats["token_lower"] = token_lower
                stats["token_upper"] = token_upper
                print(json.dumps(stats, indent=2))
                all_stats[layer][token_lower] = stats
        else:
            if args.shuffle_embeddings_per_sample:
                layer_embeddings = [sample_embedding[torch.randperm(sample_embedding.shape[0])] \
                                    for sample_embedding in layer_embeddings]
            if args.max_embeddings_per_sample:
                layer_embeddings = [sample_embedding[:args.max_embeddings_per_sample] for sample_embedding in layer_embeddings]
            layer_embeddings = torch.cat(layer_embeddings, dim=0)
            if args.shuffle_embeddings:
                layer_embeddings = layer_embeddings[torch.randperm(layer_embeddings.shape[0])]
            if args.max_embeddings:
                layer_embeddings = layer_embeddings[:args.max_embeddings]
            layer_embeddings = [layer_embeddings]
            int_dim, mles, stats = get_mle(layer_embeddings, mode=args.mle_mode, K=args.K, metric=args.metric)
            print("Layer: ", layer)
            print("Intrinsic dim: ", int_dim)
            print(json.dumps(stats, indent=2))
            all_stats[layer] = stats
    with open(os.path.join(log_folder, "stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

