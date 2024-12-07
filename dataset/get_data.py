import pandas as pd

def get_data(captions_path, data_ratio):
    train_df = pd.read_csv(f'{captions_path}train.txt', sep=',', header=None)
    train_df.head()
    train_image_ids = list(train_df[0])
    train_captions = list(train_df[1])

    # Load val images and val captions
    val_df = pd.read_csv(f'{captions_path}val.txt', sep=',', header=None)
    val_df.head()
    val_image_ids = list(val_df[0])
    val_captions = list(val_df[1])

    # Load test images and test captions
    test_df = pd.read_csv(f'{captions_path}test.txt', sep=',', header=None)
    test_df.head()
    test_image_ids = list(test_df[0])
    test_captions = list(test_df[1])


    train_size = int(len(train_captions) * data_ratio)
    val_size = int(len(val_captions) * data_ratio)
    test_size = int(len(test_captions) * data_ratio)

    print(train_size, val_size, test_size)

    train_captions = train_captions[:train_size]
    train_image_ids = train_image_ids[:train_size]

    val_captions = val_captions[:val_size]
    val_image_ids = val_image_ids[:val_size]

    test_captions = test_captions[:test_size]
    test_image_ids = test_image_ids[:test_size]

    return train_captions, train_image_ids, val_captions, val_image_ids, test_captions, test_image_ids