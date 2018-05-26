#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import hashlib
import random
import copy

def true_embeds(vocab_size, embed_size):
    return np.random.normal(0, 1, (vocab_size, embed_size))

def bucketing(cat_id, bucket_size, seed):
    value = str(cat_id) + str(seed)
    md5val = hashlib.md5(value.encode()).hexdigest()
    return int(md5val, base=16) % bucket_size

def mse(true_embeds, fitted_embeds):
    return ((true_embeds - fitted_embeds) ** 2).mean()

def fit(true_embeds, bucket_size, seeds):
    # print('fitting...')
    dims = [bucket_size for i in range(0, len(seeds))]
    coords = []
    vocab_size = true_embeds.shape[0]
    embed_size = true_embeds.shape[1]
    for i in range(0, vocab_size):
        coord = []
        for seed in seeds:
            bucket_id = bucketing(i, bucket_size, seed)
            coord.append(bucket_id)
        coord = tuple(coord)
        coords.append(coord)
    # optimize with adagrad
    factors = [np.random.normal(0, 1, (bucket_size, embed_size)) for x in range(0, len(dims))]
    alpha = 1.0
    beta = 1.0
    sum_sq_grad = [np.ones((bucket_size, embed_size)) * beta for j in range(0, len(seeds))]
    embed_ids = list(range(0, vocab_size))
    rounds = 0
    losses = [float('inf'), float('inf'), float('inf')]
    z_cnt = 0
    while True:
        rounds += 1
        loss = 0.0
        num_loop = int(math.ceil(float(100) / len(embed_ids)))
        for k in range(0, num_loop):
            random.shuffle(embed_ids)
            for i in embed_ids:
                coord = coords[i]
                # grad
                true_embed = true_embeds[i, :]
                fitted_embed = np.zeros((1, embed_size))
                for j in range(0, len(coord)):
                    fitted_embed += factors[j][(coord[j],)]
                grad = fitted_embed - true_embed
                loss += (grad ** 2).mean()
                # update
                for j in range(0, len(coord)):
                    sum_sq_grad[j][(coord[j],)] += (grad[0,] ** 2)
                    step = alpha / np.sqrt(sum_sq_grad[j][(coord[j],)])
                    factors[j][(coord[j],)] -= step * grad[0,]
        loss = loss / (num_loop * len(embed_ids))
        # print("loss: %f" % (loss,))
        # stop criterion
        losses[2] = losses[1]
        losses[1] = losses[0]
        losses[0] = loss
        if rounds > 100 and ((losses[1] - losses[0]) * (losses[2] - losses[1]) <= 0 or losses[0] / losses[1] > 0.9999 or losses[0] < 1e-20):
            z_cnt += 1
        if z_cnt > 10:
            break
    # reconstruct
    fitted_embeds = np.zeros(true_embeds.shape)
    for i in range(0, vocab_size):
        coord = coords[i]
        for j in range(0, len(coord)):
            fitted_embeds[i, :] += factors[j][(coord[j],)]
    return fitted_embeds

if __name__ == '__main__':
    embed_size = 1
    num_repeat = 10
    vocab_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    mse_results = np.zeros((num_repeat, len(vocab_sizes), 2))
    for j in range(0, len(vocab_sizes)):
        print('Fitting vocab_size = %d' % (vocab_sizes[j],))
        for i in range(0, num_repeat):
            # create random embeddings (as ground truth)
            ground_truth = true_embeds(vocab_sizes[j], embed_size)
            # fit embeddings
            seed_prefix = np.random.randint(0, 1000000)
            fitted0 = fit(ground_truth, 50, ['%d_s1' % (seed_prefix,)])  # 单桶（size = 500）
            fitted1 = fit(ground_truth, 25, ['%d_s%d' % (seed_prefix, k) for k in range(0, 2)])  # 2桶（size = 100）
            # measure mses
            mse_results[(i, j, 0)] = mse(ground_truth, fitted0)
            mse_results[(i, j, 1)] = mse(ground_truth, fitted1)
        current_mse = mse_results[:,j,:].mean(axis=0)
        line = ''
        for i in range(0, len(current_mse)):
            line += str(current_mse[i]) + ','
        print(line)