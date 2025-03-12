#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, self.data_index = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            # Handle target dictionary with both 'target' and 'biometry' keys
            if isinstance(self.next_target, dict):
                if 'target' in self.next_target:
                    self.next_target['target'] = self.next_target['target'].cuda(non_blocking=True)
                if 'biometry' in self.next_target:
                    self.next_target['biometry'] = self.next_target['biometry'].cuda(non_blocking=True)
            else:
                self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        data_index = self.data_index
        if input is not None:
            self.record_stream(input)
        if target is not None:
            # Handle recording stream for target dictionary
            if isinstance(target, dict):
                if 'target' in target:
                    target['target'].record_stream(torch.cuda.current_stream())
                if 'biometry' in target:
                    target['biometry'].record_stream(torch.cuda.current_stream())
            else:
                target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, data_index

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class DataPrefetcherCPU:
    def __init__(self, loader):
        self.loader = iter(loader)

    def preload(self):
        pass

    def next(self):
        input, target, _, data_index = next(self.loader)
        return input, target, data_index