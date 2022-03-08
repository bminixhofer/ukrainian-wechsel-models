from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass
from tqdm.auto import tqdm
import json
from pathlib import Path
import numpy as np
import math
import fasttext
from wechsel import WECHSEL
import datasets
import torch

@dataclass
class Args:
    model_name: str
    train_dir: str
    output_dir: str
    task: str
    source_word_vector_path: str
    target_word_vector_path: str
    new_tokenizer_name: str = None

def main(args):
    dataset = datasets.load_from_disk(
        args.train_dir
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # create configuration
    config = AutoConfig.from_pretrained(args.model_name)
    config.save_pretrained(output_dir)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    # train tokenizer
    if (output_dir / "tokenizer.json").exists():
        target_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    else:
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.new_tokenizer_name
            if args.new_tokenizer_name is not None
            else args.model_name
        )
        target_tokenizer = target_tokenizer.train_new_from_iterator(
            batch_iterator(), vocab_size=len(target_tokenizer)
        )
        target_tokenizer.save_pretrained(output_dir)

    source_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, add_prefix_space=False
    )

    if args.task == "clm":
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    elif args.task == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    wechsel = WECHSEL(
        fasttext.load_model(args.source_word_vector_path),
        fasttext.load_model(args.target_word_vector_path),
        bilingual_dictionary="ukrainian"
    )

    target_embeddings, info = wechsel.apply(
        source_tokenizer,
        target_tokenizer,
        model.get_input_embeddings().weight.detach().numpy(),
    )

    np.save(output_dir / "info.npy", info)
    model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = HfArgumentParser([Args])

    (args,) = parser.parse_args_into_dataclasses()
    main(args)