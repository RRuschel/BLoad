import torch.utils.data
import pickle as pkl
from PIL import Image
from pathlib import Path
import util.transforms as T
import numpy as np
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from copy import deepcopy
from typing import Callable, Optional, Union
import argparse


class AG(torch.utils.data.Dataset):
    """
    A custom dataset class for loading and preprocessing image data for object detection and 
    human-object interaction tasks. It supports reading image frames and corresponding annotations, 
    applying transformations, and preparing data for model training or evaluation.

    Attributes:
        img_set (str): Identifier for the dataset split being used (e.g., 'train' or 'val').
        img_folder (Path): Path to the directory containing image frames.
        annotations (dict): A dictionary where each key-value pair corresponds to an image or video and its annotations.
        transforms (callable): A function/transform that takes in an image and a target, and returns a transformed version.
        object_classes (list): A list of object classes, including a special '__background__' class.
        relationship_classes (list): A list of relationship classes defining interactions between objects.
        attention_relationships (list): Subset of relationship classes related to attention interactions.
        spatial_relationships (list): Subset of relationship classes related to spatial interactions.
        contacting_relationships (list): Subset of relationship classes related to contacting interactions.
        _valid_obj_ids (list): List of valid object IDs.
        _valid_verb_ids (list): List of valid verb (action) IDs.
        ids (list): A list of unique identifiers for the images or videos in the dataset.

    Parameters:
        img_set (str): The dataset split, e.g., 'train' or 'val'.
        img_folder (str): Path to the directory containing image frames.
        anno_file (str): Path to the file containing annotations.
        transforms (callable): Transformations to be applied to the images and targets.
    """

    def __init__(self, img_set: str, img_folder: Path, anno_file: Path, transforms: Optional[Callable] = None):
        print(f'Initializing Video Dataloader "{img_set}" - Reading from {img_folder}')
        self.img_set = img_set
        self.img_folder = Path(img_folder)
        with open(anno_file, 'rb') as f:
                self.annotations = pkl.load(f)
        self.transforms = transforms

        # Adding the information that the evaluator requires
        anno_path = Path(str(img_folder).replace('/frames', '/annotations'))
        self.object_classes = ['__background__']
        with open(anno_path / 'object_classes.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()

        self.relationship_classes = []
        with open(anno_path / 'relationship_classes.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

        self._valid_obj_ids = [i for i in range(37)]
        self._valid_verb_ids = [i for i in range(26)]

        self.ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: Union[int, str]):
        """
        Retrieves an item from the dataset at the specified index. The method supports indexing by
        integer (sequential access) or by string (direct access by video name).

        Parameters:
            idx (int or str): The index of the item to retrieve, which can be an integer for sequential access
                            or a string for direct access by video name.

        Returns:
            tuple: A tuple containing two lists: the first list contains the preprocessed images, and
                the second list contains the corresponding targets with annotations and additional information.
        """

        if type(idx) == int:
            video_anno = self.ids[idx]
        elif type(idx) == str:
            video_name = idx
        else:
            raise TypeError('Index type not recognized, must be int or str')

        video_anno_r = self.annotations[video_name]
        target_out = []
        img_out = []
        mask_list = []

        video_anno = deepcopy(video_anno_r)
        for img_anno in video_anno:

            frame_name = img_anno['file_name']

            mask_list.append('mask' in img_anno)

            # Add raw image to list
            img = Image.open(self.img_folder / frame_name).convert('RGB')
            w, h = img.size

            boxes = [obj['bbox'] for obj in img_anno['annotations']]
            # guard against no boxes via resizing
            boxes = np.array(boxes)  # Fix the slow warning
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

            if self.img_set == 'train':
                # Add index for confirming which boxes are kept after image transformation
                classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in
                            enumerate(img_anno['annotations'])]
            else:
                classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]

            classes = torch.tensor(classes, dtype=torch.int64)

            target = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}
            target['filename'] = img_anno['file_name']

            if self.img_set == 'train':
                boxes[:, 0::2].clamp_(min=0, max=w)
                boxes[:, 1::2].clamp_(min=0, max=h)
                keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
                boxes = boxes[keep]
                classes = classes[keep]

                target['boxes'] = boxes
                target['labels'] = classes
                target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
                target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                # Apply transform layers
                if self.transforms is not None:
                    img, target = self.transforms(img, target)

                kept_box_indices = [label[0] for label in target['labels']]

                target['labels'] = target['labels'][:, 1]

                obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
                sub_obj_pairs = []

                union_boxes = []
                w, h = img.shape[-1], img.shape[-2]

                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                        continue
                    sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                    if sub_obj_pair in sub_obj_pairs:
                        verb_labels[sub_obj_pairs.index(sub_obj_pair)][
                            self._valid_verb_ids.index(hoi['category_id'])] = 1
                    else:
                        sub_obj_pairs.append(sub_obj_pair)
                        obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                        verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                        verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                        sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                        obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                        verb_labels.append(verb_label)
                        sub_boxes.append(sub_box)
                        obj_boxes.append(obj_box)
                        union_boxes.append(self.union_box(sub_box, obj_box, h, w))
                if len(sub_obj_pairs) == 0:
                    target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                    target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                    target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['union_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                else:
                    target['obj_labels'] = torch.stack(obj_labels)
                    target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                    target['sub_boxes'] = torch.stack(sub_boxes)
                    target['obj_boxes'] = torch.stack(obj_boxes)
                    target['union_boxes'] = torch.stack(union_boxes)
            
            # Validation
            else:
                target['boxes'] = boxes
                target['labels'] = classes
                target['id'] = idx

                if self.transforms is not None:
                    img, _ = self.transforms(img, None)
                hois = []
                for hoi in img_anno['hoi_annotation']:
                    hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
                target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

            img_out.append(img)
            target_out.append(target)

        assert len(img_out) == len(target_out), f'Dimension mismatch - img: {len(img_out)} != target: {len(target_out)}'
        return img_out, target_out

    def union_box(self, sub_box: torch.Tensor, obj_box: torch.Tensor, h: int, w: int):
        """
        Calculates the union bounding box for a pair of subject and object bounding boxes.

        Parameters:
            sub_box (Tensor): The bounding box of the subject, in normalized coordinates.
            obj_box (Tensor): The bounding box of the object, in normalized coordinates.
            h (int): The height of the image.
            w (int): The width of the image.

        Returns:
            Tensor: The union bounding box of the subject and object, in normalized coordinates.
        """

        sub_box_un = sub_box * torch.tensor([w, h, w, h])
        obj_box_un = obj_box * torch.tensor([w, h, w, h])
        boxes = box_cxcywh_to_xyxy(torch.stack([sub_box_un, obj_box_un]))
        union_box = torch.stack(
            [torch.min(boxes[:, 0]), torch.min(boxes[:, 1]), torch.max(boxes[:, 2]), torch.max(boxes[:, 3])]).unsqueeze(
            0)
        union_box = box_xyxy_to_cxcywh(union_box) / torch.tensor([w, h, w, h], dtype=torch.float32)
        return union_box[0]


def make_AG_transforms(image_set: str, max_size_defined: int = 1333):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size_defined),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size_defined),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=max_size_defined),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set: str, args: argparse.Namespace):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'frames', root / 'annotations' / 'AG_HOIA_train_sgdet.pkl'),
        'val': (root / 'frames', root / 'annotations' / 'AG_HOIA_test_sgdet.pkl')
    }
    img_folder, anno_file = PATHS[image_set]
    dataset = AG(image_set, img_folder, anno_file,
                       transforms=make_AG_transforms(image_set, args.max_size_defined))
    return dataset
