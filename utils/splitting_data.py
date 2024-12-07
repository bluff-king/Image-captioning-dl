from collections import defaultdict
from sklearn.model_selection import train_test_split

# Captions file 
CAPTIONS_FILE = f'data/flickr8k/captions.txt'
with open(CAPTIONS_FILE, 'r') as f: 
    captions_data = f.readlines()[1:]

### Split dataset to 3 datasets: train (0.8), validation (0.1), test (0.1)

image_captions_dict = defaultdict(list)

for line in captions_data:
    img_id, caption = line.strip().split(',', 1)
    image_captions_dict[img_id].append(caption)
    
# Get a list of unique image IDs
unique_image_ids = list(image_captions_dict.keys())

# Split image IDs into train, val, test
train_imgs, val_test_imgs = train_test_split(unique_image_ids, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(val_test_imgs, test_size=0.5, random_state=42)

# Function to get captions and image IDs for each set
def get_captions_and_ids(image_ids_subset):
    subset_captions = []
    subset_image_ids = []
    for img_id in image_ids_subset:
        captions = image_captions_dict[img_id]
        subset_captions.extend(captions)
        subset_image_ids.extend([img_id] * len(captions))
    return subset_captions, subset_image_ids

# Get captions and image IDs for each set
train_captions, train_image_ids = get_captions_and_ids(train_imgs)
val_captions, val_image_ids = get_captions_and_ids(val_imgs)
test_captions, test_image_ids = get_captions_and_ids(test_imgs)

# Write train.txt
with open('split_data/train.txt', 'w') as f: 
    for image_id, caption in zip(train_image_ids, train_captions):
        f.write(f"{image_id},{caption}\n")

# Write val.txt
with open('split_data/val.txt', 'w') as f: 
    for image_id, caption in zip(val_image_ids, val_captions):
        f.write(f"{image_id},{caption}\n")

# Write test.txt
with open('split_data/test.txt', 'w') as f: 
    for image_id, caption in zip(test_image_ids, test_captions):
        f.write(f"{image_id},{caption}\n")
