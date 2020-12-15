import os

import torch
from fairseq.data import FairseqDataset
from torch.utils.data.dataloader import default_collate


class SignDataset(FairseqDataset):
    def __init__(self, dataset, sizes, feat_roots):
        super().__init__()
        self.dataset = dataset
        self.all_sizes = sizes

        # self.feat_roots = [root for root in feat_roots if root]
        self.feat_roots = feat_roots

    def __getitem__(self, index):
        identifier = self.dataset[index]

        all_features = []
        for level_feat_roots in self.feat_roots:
            features = [torch.cat(torch.load(os.path.join(feat_root, identifier + '.pt')), dim=0)
                        for feat_root in level_feat_roots if feat_root]
            features = torch.cat(features, dim=1)
            all_features.append(features)

        return all_features

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    @property
    def sizes(self):
        return self.all_sizes

    def num_tokens(self, index):
        raise NotImplementedError

    def size(self, index):
        return self.all_sizes[index]

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
