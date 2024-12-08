import wandb
import time
import datetime
from dataset.dataset import GloveLstmDataset, glove_lstm_collate
from dataset.get_data import get_data
from torch.utils.data import DataLoader
from models.own_embedder_lstm import ImageCaptioningLstm
from models.lr_scheduler import get_scheduler
import torch
from torch import nn

import json
with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

CAPTIONS_PATH = cfg['paths']['captions_path']
IMAGE_PATH = cfg['paths']['image_path']
lstm_params = cfg['hyperparameters']['own_embedder_lstm']
BATCH_SIZE = lstm_params['batch_size']
DATA_RATIO = lstm_params['data_ratio']
LEARNING_RATE = lstm_params['learning_rate']
WEIGHT_DECAY = lstm_params['weight_decay']
NUM_EPOCHS = lstm_params['num_epochs']
NUM_CYCLES = lstm_params['num_cycles']
CAPTIONS_LENGTH = lstm_params['captions_length']
CHECKPOINT_PATH = cfg['paths']['checkpoint_path']
EMBEDDING_DIM = lstm_params['embedding_dim']
MAX_PLATEAU_COUNT = lstm_params['max_plateau_count']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main() -> None:
    train_captions, train_image_ids, val_captions, val_image_ids, _, _ = get_data(
        CAPTIONS_PATH, DATA_RATIO
    )
    
    train_dataset = GloveLstmDataset(
        root_dir=IMAGE_PATH,
        captions=train_captions,
        image_ids=train_image_ids
    )

    val_dataset = GloveLstmDataset(
        root_dir=IMAGE_PATH,
        captions=val_captions,
        image_ids=val_image_ids
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=glove_lstm_collate,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=glove_lstm_collate,
        drop_last=True
    )
    
    model = ImageCaptioningLstm().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    lr_scheduler = get_scheduler(
        optimizer, total_steps, warmup_steps, NUM_CYCLES
    )
    
    RESUME = 'allow'
    wandb.init(
        project = "ProjectGenCap",
        resume=RESUME,
        name=str(datetime.datetime.now()),
        config={
            "model": "OwnEmbedderLstm",
            "embedding_dim": EMBEDDING_DIM,
            'num_epochs': NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "lr_scheduler": 'Cyclic',
            "optimizer" : "ADAM",
            'max_plateau_count': MAX_PLATEAU_COUNT,
            "train_size": len(train_captions),
            'val_size': len(val_captions)
            }
    )
    
    wandb.watch(model)
    
    # Initialize the best validation loss to a high value
    best_val_loss = float('inf')
    plateau_count = 0
    
    for epoch in range(NUM_EPOCHS):
        curr_lr = lr_scheduler.get_last_lr()[0]
        start_time = time.time()
        
        model.train()  # Set model to training mode
        total_train_loss = 0

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        for idx, (images, captions, next_tokens) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            next_tokens = next_tokens.to(device)

            optimizer.zero_grad()
            outputs = model(images, captions)

            loss = criterion(outputs, next_tokens)
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            total_train_loss += loss.item()

            # Print training loss at each step
            # print(f"Training Step [{idx+1}/{len(train_loader)}], Training Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss for Epoch {epoch+1}: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()  
        total_val_loss = 0

        with torch.no_grad():
            for val_idx, (images, captions, next_tokens) in enumerate(val_loader):
                images = images.to(device)
                captions = captions.to(device)
                next_tokens = next_tokens.to(device)

                outputs = model(images, captions)

                loss = criterion(outputs, next_tokens)
                total_val_loss += loss.item()

                # Print validation loss at each step
                # print(f"Validation Step [{val_idx+1}/{len(val_loader)}], Validation Loss: {loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Average Validation Loss for Epoch {epoch+1}: {avg_val_loss:.4f}")

        end_time = time.time()
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed in {(end_time - start_time)/60:.2f} minutes.")

        # Save the model if validation loss has decreased
        if avg_val_loss < best_val_loss:
            plateau_count = 0
            best_val_loss = avg_val_loss
            # print("Validation loss improved. Not saved")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f'{CHECKPOINT_PATH}own_embedder_lstm{CAPTIONS_LENGTH}.pth')
            print(f"Validation loss improved. Model checkpoint saved at epoch {epoch+1}.")
        else:
            plateau_count += 1
            print("Validation loss did not improve.")

        print('-' * 50)

        wandb.log({
            'epoch': epoch + 1,
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'starting_lr': curr_lr
        })

        if plateau_count == MAX_PLATEAU_COUNT:
            break

        model.train()  # Set model back to training mode for next epoch

    wandb.finish()

if __name__ == '__main__':
    main()
    
    