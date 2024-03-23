import argparse
import AG_BLoad
import torch
from tqdm import tqdm

# Create an argparsers that takes AG_path as string and max_size_defined as int
parser = argparse.ArgumentParser()
parser.add_argument('--AG_path', type=str, help='Path to Action Genome dataset')
parser.add_argument('--max_size_defined', type=int, default=800, help='Maximum size for the image resize augmentation')


if __name__ == '__main__':
    # Parse the arguments
    args = parser.parse_args()

    # Call the build function from AG_BLoad.py with the image_set as 'train' and the arguments
    dataset = AG_BLoad.build('train', args)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)

    for imgs, tgts, resets in tqdm(dataloader):
        pass

    print('Done')

