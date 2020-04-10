# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

from sidd.ImageLoader import ImageLoader
from sidd.MiniBatchSampler import MiniBatchSampler
from sidd.PatchAndMiniBatchSampler import PatchAndMiniBatchSampler
from sidd.PatchSampler import PatchSampler
from sidd.PatchStatsCalculator import PatchStatsCalculator
from sidd.sidd_utils import sidd_filenames_que_inst


def initialize_data_stats_queues_baselines_histograms(hps, logdir):
    # use 4 or 8 thread for faster loading
    n_thr_im = 8
    n_thr_pt = 8  # use 1 to prevent shuffling
    n_thr_mb = 8  # use 1 to prevent shuffling
    n_thr_psc = 4

    im_qsz = 32
    pat_qsz = 300
    mb_qsz_tr = 128  # hps.train_its + 1
    mb_qsz_ts = 128  # hps.test_its + 1

    requeue = True  # True: keep re-adding the same data to the queues for future epochs

    tr_fns, hps.n_tr_inst = sidd_filenames_que_inst(hps.sidd_path, 'train', hps.start_tr_im_idx, hps.end_tr_im_idx,
                                                    hps.camera, hps.iso)
    ts_fns, hps.n_ts_inst = sidd_filenames_que_inst(hps.sidd_path, 'test', hps.start_ts_im_idx, hps.end_ts_im_idx,
                                                    hps.camera, hps.iso)

    # image loaders
    tr_im_que, ts_im_que = get_image_ques(hps, requeue=requeue, n_thr_im=n_thr_im, im_qsz=im_qsz,
                                          preload=True)

    # patch samplers
    # tr_patch_sampler = PatchSampler(tr_im_que, patch_height=hps.patch_height, sampling=hps.patch_sampling,
    #                                 max_queue_size=pat_qsz, n_threads=n_thr_pt, n_reuse_image=hps.n_reuse_image,
    #                                 n_pat_per_im=hps.n_patches_per_image, shuffle=hps.shuffle_patches)
    # ts_patch_sampler = PatchSampler(ts_im_que, patch_height=hps.patch_height, sampling='uniform',
    #                                 max_queue_size=pat_qsz, n_threads=n_thr_pt, n_reuse_image=hps.n_reuse_image,
    #                                 n_pat_per_im=hps.n_patches_per_image, shuffle=hps.shuffle_patches)
    # tr_pat_que = tr_patch_sampler.get_queue()
    # ts_pat_que = ts_patch_sampler.get_queue()

    # patch stats and baselines
    if hps.calc_pat_stats_and_baselines_only:
        pat_stats = None
    else:
        pat_stats_calculator = PatchStatsCalculator(None, hps.patch_height, n_channels=4,
                                                    save_dir=logdir, file_postfix='', n_threads=n_thr_psc, hps=hps)
        pat_stats = pat_stats_calculator.load_pat_stats()
        nll_gauss, bpd_gauss = pat_stats_calculator.load_gauss_baseline()
        nll_sdn, bpd_sdn = pat_stats_calculator.load_sdn_baseline()

    # tr_batch_sampler = MiniBatchSampler(tr_pat_que, minibatch_size=hps.n_batch_train, max_queue_size=mb_qsz_tr,
    #                                     n_threads=n_thr_mb, pat_stats=pat_stats)
    # ts_batch_sampler = MiniBatchSampler(ts_pat_que, minibatch_size=hps.n_batch_test, max_queue_size=mb_qsz_ts,
    #                                     n_threads=n_thr_mb, pat_stats=pat_stats)
    tr_batch_sampler = PatchAndMiniBatchSampler(tr_im_que, minibatch_size=hps.n_batch_train, max_queue_size=mb_qsz_tr,
                                                n_threads=n_thr_mb, pat_stats=pat_stats, patch_height=hps.patch_height,
                                                sampling=hps.patch_sampling,
                                                n_pat_per_im=hps.n_patches_per_image, shuffle=hps.shuffle_patches)
    ts_batch_sampler = PatchAndMiniBatchSampler(ts_im_que, minibatch_size=hps.n_batch_test, max_queue_size=mb_qsz_ts,
                                                n_threads=n_thr_mb, pat_stats=pat_stats, patch_height=hps.patch_height,
                                                sampling=hps.patch_sampling,
                                                n_pat_per_im=hps.n_patches_per_image, shuffle=hps.shuffle_patches)

    tr_batch_que = tr_batch_sampler.get_queue()
    ts_batch_que = ts_batch_sampler.get_queue()

    # patch stats and baselines
    if hps.calc_pat_stats_and_baselines_only:
        pat_stats_calculator = PatchStatsCalculator(tr_batch_que, hps.patch_height, n_channels=4,
                                                    save_dir=logdir, file_postfix='', n_threads=n_thr_psc, hps=hps)
        pat_stats = pat_stats_calculator.calc_stats()
        nll_gauss, bpd_gauss, nll_sdn, bpd_sdn = pat_stats_calculator.calc_baselines(ts_batch_que)
        logging.trace('initialize_data_queues_and_baselines: done')

    if hps.calc_pat_stats_and_baselines_only:
        return pat_stats, nll_gauss, bpd_gauss, nll_sdn, bpd_sdn
    else:
        # return tr_im_que, ts_im_que, tr_pat_que, ts_pat_que, tr_batch_que, ts_batch_que, \
        return tr_im_que, ts_im_que, tr_batch_que, ts_batch_que, \
               pat_stats, nll_gauss, bpd_gauss, nll_sdn, bpd_sdn


def get_image_ques(hps, requeue, n_thr_im, im_qsz=16, preload=False):
    tr_fns, hps.n_tr_inst = sidd_filenames_que_inst(hps.sidd_path, 'train', hps.start_tr_im_idx, hps.end_tr_im_idx,
                                                    hps.camera, hps.iso)
    ts_fns, hps.n_ts_inst = sidd_filenames_que_inst(hps.sidd_path, 'test', hps.start_ts_im_idx, hps.end_ts_im_idx,
                                                    hps.camera, hps.iso)
    # image loaders
    tr_image_loader = ImageLoader(tr_fns, max_queue_size=im_qsz, n_threads=n_thr_im, requeue=requeue, preload=preload)
    ts_image_loader = ImageLoader(ts_fns, max_queue_size=im_qsz, n_threads=n_thr_im, requeue=requeue, preload=preload)
    tr_im_que = tr_image_loader.get_queue()
    ts_im_que = ts_image_loader.get_queue()

    return tr_im_que, ts_im_que
