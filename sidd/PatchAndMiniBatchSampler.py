# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import queue
import random
import time
from threading import Thread
import h5py
import numpy as np
from scipy.io import loadmat

from sidd.sidd_utils import pack_raw, get_nlf, load_one_tuple_images, sample_indices_uniform, \
    sidd_preprocess_standardize, sample_indices_random


class PatchAndMiniBatchSampler:

    def __init__(self, im_tuple_queue, minibatch_size=24, max_queue_size=16, n_threads=4, pat_stats=None,
                 patch_height=256, sampling='uniform', n_pat_per_im=1, shuffle=True):
        # queue to pull patch dictionaries from
        self.im_tuple_queue = im_tuple_queue
        self.mini_batch_size = minibatch_size
        self.pat_stats = pat_stats

        self.patch_height = patch_height
        self.sampling = sampling
        self.n_pat_per_im = n_pat_per_im
        self.shuffle = shuffle

        self.total_wait_time_get = 0
        self.total_wait_time_put = 0

        # initialize output queue
        self.max_queue_size = max_queue_size
        self.queue = queue.Queue(maxsize=self.max_queue_size)

        # initialize threads
        self.threads = []
        self.n_threads = n_threads
        for t in range(self.n_threads):
            self.threads.append(
                Thread(target=self.sample_minibatch_thread, args=[t]))
            self.threads[t].start()

    def sample_minibatch_thread(self, thread_id):
        mb_cnt = 0
        # first image, first patch generator
        im_tuple = self.im_tuple_queue.get()
        patch_gen = self.patch_generator(im_tuple)
        while True:
            # Dequeue patch dictionaries into mini-batch
            mini_batch_x = None
            mini_batch_y = None
            mini_batch_pid = None
            pat_dict = None
            for p in range(self.mini_batch_size):
                try:
                    pat_dict = next(patch_gen)
                except StopIteration:
                    im_tuple = self.im_tuple_queue.get()
                    patch_gen = self.patch_generator(im_tuple)
                    pat_dict = next(patch_gen)
                if p == 0:
                    p_shape = pat_dict['in'].shape
                    mini_batch_x = np.zeros((self.mini_batch_size, p_shape[1], p_shape[2], p_shape[3]))
                    mini_batch_y = np.zeros((self.mini_batch_size, p_shape[1], p_shape[2], p_shape[3]))
                    mini_batch_pid = np.zeros(self.mini_batch_size)
                mini_batch_x[p, :, :, :] = pat_dict['in']
                mini_batch_y[p, :, :, :] = pat_dict['gt']
                mini_batch_pid[p] = pat_dict['pid']  # patch index in image
            # only one value for the whole mini-batch:
            mini_batch_nlf0 = [pat_dict['nlf0']]
            mini_batch_nlf1 = [pat_dict['nlf1']]
            mini_batch_iso = [pat_dict['iso']]
            mini_batch_cam = [pat_dict['cam']]

            self.queue.put({'_x': mini_batch_x, '_y': mini_batch_y, 'pid': mini_batch_pid,
                            'nlf0': mini_batch_nlf0, 'nlf1': mini_batch_nlf1,
                            'iso': mini_batch_iso, 'cam': mini_batch_cam,
                            'fn': pat_dict['fn'], 'metadata': pat_dict['metadata']})
            mb_cnt += 1
            # if self.patch_tuple_queue.empty():
            #     print('patch queue empty, # sampled minibatches = %d' % mb_cnt)

    def get_queue(self):
        return self.queue

    def get_total_wait_time(self):
        return self.total_wait_time_get, self.total_wait_time_put

    def patch_generator(self, im_tuple):
        H = im_tuple['in'].shape[1]
        W = im_tuple['in'].shape[2]
        if self.sampling == 'uniform':  # use all patches in image
            ii, jj, n_p = sample_indices_uniform(H, W, self.patch_height, self.patch_height, shuf=self.shuffle,
                                                 n_pat_per_im=self.n_pat_per_im)
            if n_p != self.n_pat_per_im:
                print('# patches/image = %d != %d' % (n_p, self.n_pat_per_im))
                print('fn = %s' % str(im_tuple['fn']))
                import pdb
                pdb.set_trace()
        else:  # use self.n_pat_per_im patches
            ii, jj = sample_indices_random(H, W, self.patch_height, self.patch_height, self.n_pat_per_im)
        pid = 0
        for (i, j) in zip(ii, jj):
            in_patch = im_tuple['in'][:, i:i + self.patch_height, j:j + self.patch_height, :]
            gt_patch = im_tuple['gt'][:, i:i + self.patch_height, j:j + self.patch_height, :]
            pat_dict = {'in': in_patch, 'gt': gt_patch, 'vr': [], 'nlf0': im_tuple['nlf0'],
                        'nlf1': im_tuple['nlf1'], 'iso': im_tuple['iso'], 'cam': im_tuple['cam'],
                        'fn': im_tuple['fn'], 'metadata': im_tuple['metadata'], 'pid': pid}
            pid += 1
            yield pat_dict
