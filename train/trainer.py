import torch
from tqdm import tqdm, trange
from typing import Optional
from lion_pytorch import Lion
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dataset.CapDataset import CapDataset

class Trainer():
    def __init__(
            self,
            args,
            model: Optional[torch.nn.Module],
            train_dataset: Optional[CapDataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

    def train(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.args.train_batch_soze,
            drop_last = self.args.dataloader_drop_last,
            
        )

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = (
                len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
            )

        optimizer = self.get_optimizer()
        scheduler = get_scheduler(
            self.args.learning_rate,
            optimizer = optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc = "Epoch")
        
        scaler = torch.cuda.amp.GradScaler()

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc = "Iteration", position = 0, leave = True
            )
            
            for step,batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "is_train": True
                }

                with torch.cuda.amp.autocast():
                    loss = self.model(**inputs)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss/self.args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                tr_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()

                    self.model.zero_grad()
                    global_step += 1

        return
    
    def get_optimizer(self):
        decay_paprameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_paprameters = [name for name in decay_paprameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_paprameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters if n not in decay_paprameters
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = Lion(
            optimizer_grouped_parameters,
            lr = self.args.learning_rate,
        )

        return optimizer