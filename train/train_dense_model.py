import torch
import argparse
from transformers import set_seed, RobertaConfig, AutoTokenizer
from models.dense_model import DenseModel
from dataset.CapDataset import CapDataset
from trainer import Trainer

MODEL_CLASSES_TRAIN = {
    'sentence-transformers': (RobertaConfig, DenseModel, AutoTokenizer)
}

MODEL_PATH_MAP = {
    'sentence-transformers': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
}

def load_tokenizer(args):
    return MODEL_CLASSES_TRAIN[args.model_type][2].from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer
    )

def main(args):
    set_seed(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = load_tokenizer(args) 
    config_class, model_class, _ = MODEL_CLASSES_TRAIN[args.model_type]

    if args.pretrained:
        model = model_class.from_pretrained(
            args.pretrained_path,
            torch_dtype = args.compute_dtype,
            device_map = args.device,
            args = args,
        )
    else:
        model_config = config_class.from_pretrained(
            args.model_name_or_path, finetuning_task = args.token_level
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            torch_dtype = args.compute_dtype,
            config = model_config,
            device_map = args.device,
            args = args,
        )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    

    train_dataset = CapDataset(args, tokenizer, "train")

    trainer = Trainer(
        args = args,
        model = model,
        train_dataset = train_dataset,
        tokenizer = tokenizer,
    )

    if args.do_train:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_dir',
        default=None,
        required=True,
        type = str,
        help="Path to save or load model",
    )

    parser.add_argument(
        "--data_dir",
        default = None,
        type = str,
        help = "the input data directory",
    )

    parser.add_argument(
        "--token_level",
        type = str,
        default = "word-level",
    )

    parser.add_argument(
        "--model_type",
        default = 'roberta',
        type = str
    )

    parser.add_argument(
        "--do_train",
        action = 'store_true',
    )

    parser.add_argument(
        "--pretrained",
        action = "store_true"
    )

    parser.add_argument(
        "--pretrained_path",
        default = None,
        type = str
    )

    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
    )

    parser.add_argument(
        "--num_train_epochs",
        default = 20.0,
        type = float,
    )

    parser.add_argument(
        "--train_batch_size",
        default = 64,
        type = int,
    )

    parser.add_argument(
        "--dataloader_drop_last",
        default = True,
        type = bool,
    )

    parser.add_argument(
        "--max_seq_len_query",
        default = 256,
        type = int,
    )

    parser.add_argument(
        "--use_fast_tokenizer",
        default = False,
        type = bool,
    )

    #Optimizer
    parser.add_argument(
        "--gradient_checkpointing",
        action = "store_true",
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type = float,
    )

    parser.add_argument(
        "--lr_scheduler_type",
        default = "cosine",
        type = str,
    )

    parser.add_argument(
        "--weight_decay",
        default = 0.0,
        type = float,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type = int,
        default = 1,
    )

    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float, 
    )
    parser.add_argument(
        "--warmup_steps", 
        default=100, 
        type=int, 
    )
    parser.add_argument(
        "--max_steps",
        default=-1, #no limit
        type=int,
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
    )

    #Model
    parser.add_argument(
        "--compute_dtype",
        type=torch.dtype,
        default=torch.float,
    )
    parser.add_argument(
        "--pooler_type",
        default="avg",
        type=str,
    )
    parser.add_argument(
        "--sim_fn",
        default="dot",
        type=str,
    )
    parser.add_argument(
        "--label_smoothing",
        default=0.00,
        type=float,
    )

    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]

    main(args)

