import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle
import json_lines

from pytorch_pretrained_bert.tokenization import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader
import pdb
import csv
import sys

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _converId(img_id):

    img_id = img_id.split('-')
    if 'train' in img_id[0]:
        new_id = int(img_id[1])
    elif 'val' in img_id[0]:
        new_id = int(img_id[1]) + 1000000        
    elif 'test' in img_id[0]:
        new_id = int(img_id[1]) + 2000000    
    else:
        pdb.set_trace()

    return new_id


def _load_annotationsVGSG_R(annotations_jsonpath, split):
    entries = []
    with open(annotations_jsonpath, 'r') as f:
        scene_graphs = json.load(f)
        if split=='train':
            scene_graphs = scene_graphs[:int(0.9*len(scene_graphs))]
        elif split=='val':
            scene_graphs = scene_graphs[int(0.9*len(scene_graphs)):]
        for scene_graph in scene_graphs:
            if split == 'test':
                pass
            else:
                objects = scene_graph['objects']
                if len(objects)==0:
                    continue
                objects2name = {x['object_id']:(x['names'][0], x['synsets'][0] if len(x['synsets'])>0 else -1) for x in objects}
                object_list = list(objects2name.values())
                relationships = scene_graph['relationships']
                relation_tuples = [(objects2name[x["subject_id"]], (x['predicate'], x['synsets'][0] if len(x['synsets'])>0 else -1), objects2name[x["object_id"]]) for x in relationships]
                num_obj = len(objects)
                num_rel = len(relation_tuples)
                # filter out phrase relation
                filtered_relation_tuples = [] 
                for rel in relation_tuples:
                    if len(rel[1][0].split())==1 and len(rel[0][0].split())==1 and len(rel[2][0].split())==1:
                        filtered_relation_tuples.append(rel)
                entries.append(
                    {"image_id":scene_graph['image_id'], 'relations': filtered_relation_tuples, 'objects': object_list}
                )

    return entries

class VGSGNoVisDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
        max_seq_length: int = 40,
        max_region_num: int = 60
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        if task == 'VGenomeSceneGraph':
            self._entries = _load_annotationsVGSG_R(annotations_jsonpath, split)
        else:
            assert False
        self._split = split
        self._image_features_reader = image_features_reader
        self._gt_image_features_reader = gt_image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self._max_region_num = max_region_num
        self.num_labels = 1

        self._names = []
        if not os.path.exists(os.path.join(dataroot, "cache")):
            os.makedirs(os.path.join(dataroot, "cache"))

        # cache file path data/cache/train_ques
        cache_path = "data/VGSG/cache/" + split + '_' + task + "_" + str(max_seq_length) + "_" + str(max_region_num) + "_vcr.pkl"
        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._entries, open(cache_path, 'wb'))
        else:
            self._entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        count = 0
        for entry in self._entries:
            token_pairs = []
            for relation in entry['relations']:
                assert len(relation) == 3
                token_pairs.append((relation[0][0],relation[1][0],relation[2][0]))

            num_rels = len(entry['relations'])
            num_random_rels = (self._max_seq_length - 2) // 3 - num_rels

            if num_random_rels>0:
                pass
                # gt_pairs = {(rel[0],rel[2]) for rel in entry['relations']}
                # random_pairs = self._get_random_pair(entry['objects'], gt_pairs, num_random_rels)
                # for pair in list(random_pairs):
                #     token_pairs.append((pair[0][0],'background', pair[1][0]))
            else:
                for i in range(-num_random_rels):
                    token_pairs.pop()

            random.shuffle(token_pairs)
            tokens = []
            for pair in token_pairs:
                tokens.extend(pair)

            tokens = ['[CLS]'] + tokens + ['[SEP]']
            tokens_char = tokens

            target = [self._tokenizer.vocab.get(self._tokenizer.tokenize(x)[0], self._tokenizer.vocab['[UNK]']) if i%3==2 else -1 for i, x in enumerate(tokens)]
            tokens = [self._tokenizer.vocab.get(self._tokenizer.tokenize(x)[0], self._tokenizer.vocab['[UNK]']) if i%3!=2 else self._tokenizer.vocab.get('[MASK]', self._tokenizer.vocab['[UNK]']) for i, x in enumerate(tokens)]
            
            for i in range(len(tokens)):
                if target[i] != -1:
                    print(tokens_char[i],tokens[i],target[i])

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)
            # input_mask = [1 if i%3==2 else 0 for i in range(len(tokens))]
            # co_attention_mask = [-1 if i%3==2 else 1 for i in range(len(tokens))]
            # co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
            # co_attention_mask[0] = -1
            # co_attention_mask[-1] = -1
                
            if len(tokens) < self._max_seq_length:
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding 
                target += [-1] * len(padding)  

            assert_eq(len(tokens), self._max_seq_length)
            entry['input_ids'] = tokens 
            entry["input_mask"] = input_mask
            entry['segment_ids'] = segment_ids
            # entry["co_attention_mask"] = co_attention_mask
            entry['target'] = target

            sys.stdout.write('%d/%d\r' % (count, len(self._entries)))
            sys.stdout.flush()
            count += 1

    def tensorize(self):

        for entry in self._entries:
            input_ids = torch.from_numpy(np.array(entry["input_ids"]))
            entry["input_ids"] = input_ids

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

            target = torch.from_numpy(np.array(entry["target"]))
            entry["target"] = target

    def _get_random_pair(self, object_list, gt_pairs, num_pairs):
        num_obj = len(object_list)
        candidate_pair_set = set()
        for i in range(num_obj):
            for j in range(i,num_obj):
                candidate_pair_set.add((object_list[i], object_list[j]))
                candidate_pair_set.add((object_list[j], object_list[i]))
        candidate_pair_set = candidate_pair_set - gt_pairs
        return random.choices(list(candidate_pair_set),k=min(num_pairs, len(candidate_pair_set)))

    def __getitem__(self, index):
        
        entry = self._entries[index]

        image_id = entry["image_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        gt_features, gt_num_boxes, gt_boxes, _ = self._gt_image_features_reader[image_id]

        # merge two features.
        features[0] = (features[0] * num_boxes + gt_features[0] * gt_num_boxes) / (num_boxes + gt_num_boxes)

        # merge two boxes, and assign the labels. 
        gt_boxes = gt_boxes[1:gt_num_boxes]
        gt_features = gt_features[1:gt_num_boxes]
        gt_num_boxes = gt_num_boxes - 1

        gt_box_preserve = min(self._max_region_num-1, gt_num_boxes)
        gt_boxes = gt_boxes[:gt_box_preserve]
        gt_features = gt_features[:gt_box_preserve]
        gt_num_boxes = gt_box_preserve
 
        num_box_preserve = min(self._max_region_num - int(gt_num_boxes), int(num_boxes))
        boxes = boxes[:num_box_preserve]
        features = features[:num_box_preserve]

        # concatenate the boxes
        mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
        mix_features = np.concatenate((features, gt_features), axis=0)
        mix_num_boxes = num_box_preserve + int(gt_num_boxes)
        
        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        # mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        # mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        input_ids = entry["input_ids"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        target = entry["target"]

        assert_eq(len(input_ids),len(input_mask))
        assert_eq(len(input_mask),len(segment_ids))
        assert_eq(len(segment_ids),len(target))

        if self._split == 'test':
            # anno_id = entry["anno_id"]
            anno_id = 0#entry["anno_id"]
        else:
            anno_id = entry["image_id"]

        co_attention_mask = torch.zeros((1, self._max_region_num, self._max_seq_length))
        input_ids = input_ids.unsqueeze(1)
        input_mask = input_mask.unsqueeze(1)
        segment_ids = segment_ids.unsqueeze(1)
        return features, spatials, image_mask, input_ids, target, input_mask, segment_ids, co_attention_mask, anno_id

    def __len__(self):
        return len(self._entries)

    def get_tokenizer(self):
        return self._tokenizer
