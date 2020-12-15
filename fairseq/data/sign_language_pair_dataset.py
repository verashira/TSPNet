# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from . import data_utils, FairseqDataset

logger = logging.getLogger(__name__)


def create_adjacent_matrix(multi_src_tokens):
    """
    src_tokens: list of Tensors, each corresponds to one level of features in a batch
    """
    num_nodes = [src_tokens.shape[1] for src_tokens in multi_src_tokens]
    starting_indices = [0] + np.cumsum(num_nodes)[:-1].tolist()

    sum_num_nodes = sum(num_nodes)
    A = torch.eye(sum_num_nodes)
    for lv in range(1, len(num_nodes)):
        for i in range(num_nodes[lv]):
            link_start = starting_indices[lv] + i  
            for j in range((16 - 8) // 2 + 1):
                link_end = starting_indices[lv - 1] + i + j
                if i + j >= num_nodes[lv - 1]:
                    break

                A[link_start, link_end] = 1
                A[link_end, link_start] = 1

    # same-level self attention
    for lv in range(0, len(num_nodes)):
        for i in range(num_nodes[lv]):
            for j in range(num_nodes[lv]):
                A[starting_indices[lv] + i, starting_indices[lv] + j] = 1

    return A


def create_adjacent_matrix_multilv(multi_src_tokens, span_lengths, level_links,
                                   stride, eye=True, same_level_links=True, symmetric=False):
    """
    Args:
        src_tokens: list of Tensors, each corresponds to one level of features in a batch
        span_lengths: the length of span each level of feature corresponds to
        level_links: list of tuple of (lv, lv) to build up connections
        stride: stride used when extracting features from span
        eye: if to initialize A with eye matrix
        symmetric: if to make result matrix symmetric
    Returns:
        A: adjacent matrix
    Note:
        Notice the result adjacent matrix should be viewed as (Target x Source),
        and each tuple (link_level_1, link_level_2) corresponds to part of the matrix as:
             | (0, 0) | (0, 1) | (0, 2) | ...
             | (1, 0) | (1, 1) | (1, 2) | ...
             | (2, 0) | (2, 1) | (2, 2) | ...
    """
    num_nodes = [src_tokens.shape[1] for src_tokens in multi_src_tokens]
    start_indices = [0] + np.cumsum(num_nodes)[:-1].tolist()

    total_num_nodes = sum(num_nodes)
    if eye:
        A = torch.eye(total_num_nodes)
    else:
        A = torch.zeros((total_num_nodes, total_num_nodes))

    for link_level_1, link_level_2 in level_links:
        A_part = A[
                 start_indices[link_level_1]: start_indices[link_level_1] + num_nodes[link_level_1],
                 start_indices[link_level_2]: start_indices[link_level_2] + num_nodes[link_level_2]
                 ]

        if symmetric:
            A_part_sym = A[
                         start_indices[link_level_2]: start_indices[link_level_2] + num_nodes[link_level_2],
                         start_indices[link_level_1]: start_indices[link_level_1] + num_nodes[link_level_1]
                         ]

        span1 = span_lengths[link_level_1]
        span2 = span_lengths[link_level_2]

        for i in range(num_nodes[link_level_1]):
            span_coverage = abs(span1 - span2) // stride + 1
            start, end = i, min(i + span_coverage, num_nodes[link_level_2])
            A_part[i, start:end] = 1

            if symmetric:
                A_part_sym[start:end, i] = 1

    # same-level self attention
    if same_level_links:
        for lv in range(0, len(num_nodes)):
            A[
            start_indices[lv]: start_indices[lv] + num_nodes[lv],
            start_indices[lv]: start_indices[lv] + num_nodes[lv]
            ] = 1

    return A


def add_candidates_selflinks_multilv(As, src_tokens, span_lengths, stride, eye=True):
    self_links = create_adjacent_matrix_multilv(src_tokens,
                                                span_lengths=span_lengths,
                                                level_links=[],
                                                stride=stride,
                                                eye=eye,
                                                same_level_links=True,
                                                symmetric=True,
                                                )

    num_levels = len(src_tokens)

    start = 0
    end = src_tokens[0].shape[1]
    for lv in range(num_levels):
        mask = torch.zeros_like(self_links)
        if lv > 0:
            start += src_tokens[lv - 1].shape[1]
            end += src_tokens[lv].shape[1]

        mask[start:end, start:end] = 1

        As.append(self_links * mask)


def add_candidates_crosslinks_multilv(As, src_tokens, span_lengths, level_links, stride, symmetric):
    cross_links = create_adjacent_matrix_multilv(src_tokens,
                                                 span_lengths=span_lengths,
                                                 level_links=level_links,
                                                 stride=stride,
                                                 eye=False,
                                                 same_level_links=False,
                                                 symmetric=symmetric
                                                 )

    num_levels = len(src_tokens)

    num_nodes = [src_tokens.shape[1] for src_tokens in src_tokens]
    start_indices = [0] + np.cumsum(num_nodes)[:-1].tolist()

    for lv1 in range(num_levels):
        for lv2 in range(num_levels):
            if lv1 == lv2:
                continue

            row_start, row_end = start_indices[lv1], start_indices[lv1] + num_nodes[lv1]
            col_start, col_end = start_indices[lv2], start_indices[lv2] + num_nodes[lv2]

            mask = torch.zeros_like(cross_links)
            mask[row_start:row_end, col_start:col_end] = 1

            As.append(cross_links * mask)


def collate_signs(values, eos_idx=None, left_pad=False):
    """Convert a list of 2d tensors into a padded 3d tensor.

    input is a list of k_i * 1024 features.
    We need to find the max K and pad all others to have the same length.
    Return NUM_SAMPLES * K * 1024 tensor.
    """

    # v is a k x 1024 vectors.
    lens = [len(v) for v in values]
    max_len = max(lens)
    min_len = min(lens)

    assert min_len > 0

    res = torch.zeros(size=(len(values), max_len, values[0].shape[1]))

    for i, v in enumerate(values):
        if left_pad:
            res[i][max_len - len(v):] = v
        else:
            res[i][:len(v)] = v

    return res



def collate(
        samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True, multilv_args=None
):
    if len(samples) == 0:
        return {}

    def merge_tokens(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_signs(key, left_pad):
        return collate_signs(
            [s[key] for s in samples], left_pad
        )

    def build_src_tokens(samples):
        num_levels = len(samples[0]['source'])

        multi_src_tokens, multi_src_lengths, multi_encoder_padding_mask = [], [], []
        for lv in range(num_levels):
            src_lengths = torch.LongTensor([len(s['source'][lv]) for s in samples])
            # TODO: sort_order is assumed to be consistent among different feature levels, may need to verify this
            src_lengths, sort_order = src_lengths.sort(descending=True)
            src_tokens = collate_signs([s['source'][lv] for s in samples], left_pad_source)
            src_tokens = src_tokens.index_select(0, sort_order)
            encoder_padding_mask = torch.arange(src_lengths.max())[None, :] > src_lengths[:, None]
            multi_src_tokens.append(src_tokens)
            multi_src_lengths.append(src_lengths)
            multi_encoder_padding_mask.append(encoder_padding_mask)

        prev_output_tokens = None

        tgt_tokens = merge_tokens('target', left_pad=left_pad_target)
        tgt_tokens = tgt_tokens.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge_tokens(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

        return ntokens, multi_src_tokens, multi_src_lengths, multi_encoder_padding_mask, tgt_tokens, prev_output_tokens, sort_order

    id = torch.LongTensor([s['id'] for s in samples])

    ntokens, src_tokens, src_lengths, encoder_padding_mask, \
    tgt_tokens, prev_output_tokens, sort_order = build_src_tokens(samples)


    adjacent_matrix = create_adjacent_matrix_multilv(src_tokens,
                                                     span_lengths=multilv_args['span_lengths'],
                                                     level_links=multilv_args['level_links'],
                                                     stride=multilv_args['stride'],
                                                     eye=multilv_args['eye'],
                                                     same_level_links=multilv_args['same_level_links'],
                                                     symmetric=multilv_args['symmetric']
                                                     )

    A_matrix = []

    add_candidates_selflinks_multilv(A_matrix,
                                     src_tokens,
                                     span_lengths=multilv_args['span_lengths'],
                                     stride=multilv_args['stride'],
                                     eye=multilv_args['stride']
                                     )

    if len(src_tokens) > 1:
        # 2. cross-level links
        add_candidates_crosslinks_multilv(A_matrix,
                                          src_tokens,
                                          span_lengths=multilv_args['span_lengths'],
                                          level_links=multilv_args['level_links'],
                                          stride=multilv_args['stride'],
                                          symmetric=multilv_args['symmetric']
                                          )


    A_matrix = torch.cat([Am.unsqueeze(-1) for Am in A_matrix], dim=-1)

    id = id.index_select(0, sort_order)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'encoder_padding_mask': encoder_padding_mask,
            'adjacent_matrix': adjacent_matrix,
            'candidate_matrices': A_matrix,
        },
        'target': tgt_tokens
    }

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class SignLanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
            self, src, src_sizes,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True,
            remove_eos_from_source=False, append_eos_to_target=False,
            align_dataset=None,
            append_bos=False,
            use_bucketing=True,
            multilv_args=None
    ):
        self.src = src
        self.tgt = tgt

        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None

        self.tgt_dict = tgt_dict

        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos

        self.use_bucketing = use_bucketing

        self.multilv_args = multilv_args

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }
        # if self.align_dataset is not None:
        #     example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.tgt_dict.pad(), eos_idx=self.tgt_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            multilv_args=self.multilv_args
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if not self.use_bucketing:
                return indices
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]

        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
