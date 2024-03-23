from AG import AG, make_AG_transforms
from collections import defaultdict
import copy
import random
from pathlib import Path
from typing import Optional

class AG_BLoad(AG):
    """
    An extension of the AG dataset class designed to handle video datasets where videos have varying lengths. 
    This class introduces a method to group videos into blocks of similar frame counts to optimize batch loading
    and processing. It supports dynamic randomization of these blocks to ensure diversity in training and evaluation.

    Inherits:
        AG: The base dataset class providing fundamental dataset loading and preprocessing functionalities.

    Attributes inherited from AG:
        img_set, img_folder, annotations, transforms, object_classes, relationship_classes,
        attention_relationships, spatial_relationships, contacting_relationships, _valid_obj_ids,
        _valid_verb_ids, ids

    New Attributes:
        n_frames_per_video (defaultdict): A dictionary mapping the number of frames to video names, used for block generation.
        blocks (list): A list of lists, where each sublist represents a block of videos grouped by similar frame count.
        resets (list): A list of reset points within each block to handle video boundaries.
        block_size (int): The size of the largest block, determining the uniform length for all blocks through padding.
    """

    def __init__(self, img_set, img_folder, anno_file, transforms):
        super().__init__(img_set, img_folder, anno_file, transforms)
        
        # To be able to generate the blocks, we needs to know which videos have the same size
        # Will need to be slightly modified if we want to use this for another dataset
        n_frames_per_video = defaultdict(lambda: [])
        for video_name in self.ids:
            n_frames_per_video[len(self.annotations[video_name])].append(video_name)

        self.n_frames_per_video = n_frames_per_video

        # Generate the blocks - Will need to be called at the end of each epoch for further randomization
        self.randomize()
        print(f'Generated {len(self.blocks)} blocks of size {self.block_size} for {img_set} set')
        original_n_frames = sum([len(self.annotations[video_name]) for video_name in self.ids])
        print(f'Number of frames = {len(self)*self.block_size} - Original number of frames = {original_n_frames}')

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx: int):
        """
        Retrieves a block of videos from the dataset at the specified index. Each block is composed of one or more videos
        that are concatenated together along with their annotations. If the total frame count of a block is less than the
        maximum block size, it is padded with the last frame to match the block size.

        Parameters:
            idx (int): The index of the block to retrieve.

        Returns:
            tuple: A tuple containing three lists: the first list contains the preprocessed frames for the entire block,
                the second list contains the corresponding targets for each frame, and the third list contains reset
                indices indicating the start of a new video within the block.
        """

        video_list = []
        targets_list = []
        resets_list = []

        for video_name, resets in zip(self.blocks[idx], self.resets[idx]):
            frames, targets = super().__getitem__(video_name)
            video_list.extend(frames)
            targets_list.extend(targets)
            if type(resets) == int:
                # Handles the largest video which won't have a reset
                resets = [resets]
            resets_list.extend(resets)

        # Rather often we will have a block that is not the same size as the block size
        # So we need to pad the video_list and targets_list
        # For simplicity, a simple repetition of the last frame will be used
        
        while len(video_list) < self.block_size:
            video_list.append(video_list[-1])
            targets_list.append(targets_list[-1])

        assert len(video_list) == self.block_size, f'video_list size: {len(video_list)} - block_size: {self.block_size} - targets_list size: {len(targets_list)}'

        return video_list, targets_list, resets_list
    

    def collate_fn(self, batch: list):
        """
        A custom collate function for batch processing. It groups the data collected by the DataLoader into batches.

        Parameters:
            batch (list): A list of tuples, where each tuple contains video frames, targets, and reset indices from a single block.

        Returns:
            tuple: A batched tuple of video frames, targets, and reset indices suitable for processing by a model.
        """

        return tuple(zip(*batch))


    def randomize(self, seed: Optional[int] = 0):
        """
        Randomizes the organization of videos into blocks based on their frame counts. This method ensures diversity
        in training and evaluation by changing the composition of blocks every epoch, based on the specified seed.

        Parameters:
            seed (int, optional): A seed value for the random number generator to ensure reproducibility. Defaults to 0.
        """

        video_frames_count_dict = copy.deepcopy(self.n_frames_per_video)

        frame_count_options = sorted(list(video_frames_count_dict.keys()))
        largest_block_size = max(frame_count_options)
        blocks = []
        block_reset_indices = []
        rng = random.Random(seed * 100)  ## fixing random selection for all processes ##
        while len(video_frames_count_dict) > 0:
            current_block = []
            current_reset_indices = [0]
            remaining_block_size = largest_block_size
            while remaining_block_size > 0 and len(frame_count_options) > 0:
                if frame_count_options[0] > remaining_block_size:  # We can't add more videos
                    break
                else:
                    allowed_max_size = min(remaining_block_size, frame_count_options[-1])
                    size_differences = [(count_option - allowed_max_size) for count_option in frame_count_options]
                    size_differences = [diff for diff in size_differences if diff <= 0]
                    closest_size_index = frame_count_options.index(frame_count_options[size_differences.index(max(size_differences))])

                    chosen_size_index = rng.randint(0, closest_size_index)
                    chosen_frame_count = frame_count_options[chosen_size_index]
                    chosen_video_index = rng.randint(0, len(video_frames_count_dict[chosen_frame_count]))
                    if chosen_video_index > 0:
                        chosen_video_index -= 1
                        # Note: Adjusting index for random selection.
                    selected_video_id = video_frames_count_dict[chosen_frame_count][chosen_video_index]
                    current_block.append(selected_video_id)
                    current_reset_indices.append(current_reset_indices[-1] + chosen_frame_count)
                    remaining_block_size -= chosen_frame_count

                    video_frames_count_dict[chosen_frame_count].remove(selected_video_id)
                    if len(video_frames_count_dict[chosen_frame_count]) == 0:
                        video_frames_count_dict.pop(chosen_frame_count)
                        frame_count_options.remove(chosen_frame_count)

            blocks.append(current_block)
            block_reset_indices.append(current_reset_indices[0:-1])
            assert len(blocks[-1]) == len(block_reset_indices[-1])
            block_reset_indices[-1].append(largest_block_size)
        
        self.blocks = blocks
        self.resets = block_reset_indices
        self.block_size = largest_block_size


def build(image_set, args):
    root = Path(args.AG_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'frames', root / 'annotations' / 'AG_HOIA_train_sgdet.pkl'),
        'val': (root / 'frames', root / 'annotations' / 'AG_HOIA_test_sgdet.pkl')
        # 'val': (root / 'frames', root / 'annotations' / 'AG_HOIA_train_sgdet.pkl')
    }
    img_folder, anno_file = PATHS[image_set]
    dataset = AG_BLoad(image_set, img_folder, anno_file,
                       transforms=make_AG_transforms(image_set, args.max_size_defined))
    return dataset