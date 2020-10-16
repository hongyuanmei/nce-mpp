import os 
import psutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#process = psutil.Process(os.getpid())

"""
draw Monte-Carlo samples for GNHP
"""

def draw_mc_samples(c, cb, d, o, dtime, M, device): 
    """
    input
        c [B x D] : c_i in NHP
        cb [B x D] : \bar{c}_i in NHP
        d [B x D] : d_i in NHP, how fast c(t) goes from c_i to \bar{c}_i
        o [B x D] :  o_i in NHP
        dtime [B] : actual time intervals between last and next actual event 
        M [B] : # of MC samples per interval of each sequence in batch
    """
    #print(f"start sampling MC samples")
    #print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')
    M_max = torch.max(M)
    batch_size, D = c.size()
    u = torch.ones(
        size=[batch_size, M_max], dtype=torch.float32, device=device)
    u = u.uniform_(0.0, 1.0)
    dtime_inter = dtime.unsqueeze(-1)
    dtime_inter = dtime_inter * u # uniformly sample dtimes over interval
    
    c_inter = c.unsqueeze(1).expand(batch_size, M_max, D)
    cb_inter = cb.unsqueeze(1).expand(batch_size, M_max, D)
    d_inter = d.unsqueeze(1).expand(batch_size, M_max, D)
    o_inter = o.unsqueeze(1).expand(batch_size, M_max, D)

    """
    mask out the padded samples 
    overhead for the padded elements can't be avoided
    """
    mask_inter = torch.ones(
        size=[batch_size, M_max + 1], dtype=torch.float32, device=device)
    # init as all 1.0
    r = torch.arange(0, batch_size, dtype=torch.long, device=device)
    mask_inter[r, M] = 0.0 # for each seq, set 0.0 at its length
    mask_inter = mask_inter.cumprod(dim=1) # roll 0.0 over to the end
    mask_inter = mask_inter[:, :-1]
    #print(f"after sampling MC samples")
    #print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')
    return c_inter, cb_inter, d_inter, o_inter, dtime_inter, mask_inter

"""
draw random types for all point processes
"""

def sample_noise_types(probs, s0, s1, num, event_num, noise_mode, device): 
    """
    input 
        probs [tensor of s0 x s1 x K] : probabilities over K types
        s0 [int] : size-0 of probs 
        s1 [int] : size-1 of probs 
        num [int] : # of i.i.d. samples per probs[i,j]
    """
    """
    NOTE : sampling from multinomial is slow
    SOLUTION : we sample case by case
    we can choose to always sample uniformly : cuz it is fast
    otherwise
        if it is on CPU : 
            convert it to numpy and do multinomial sampling there
        if it is on GPU : 
            use torch.multinomial()
    """
    if noise_mode == 'uniform': 
        rst = sample_uniform(s0, s1, num, event_num)
    elif noise_mode == 'multinomial': 
        if device.type == 'cpu': 
            rst = sample_multinomial_numpy(probs, s0, s1, num, event_num)
        elif device.type == 'cuda': 
            rst = sample_multinomial_torch(probs, s0, s1, num, event_num)
        else: 
            raise Exception(f"Unknown device type : {device.type}")
    else: 
        raise Exception(f"Unknow noise mode : {noise_mode}")
    return rst

def sample_multinomial_torch(probs, s0, s1, num, event_num): 
    # probs are unnormalized 
    probs_view = probs.view(-1, event_num)
    rst = torch.multinomial(
        probs_view, num, replacement=True)
    rst = rst.view(s0, s1, num) # s0 x s1 x NUM
    return rst

def sample_multinomial_numpy(probs, s0, s1, num, event_num): 
    # probs are unnormalized 
    probs_view = probs.view(-1, event_num) # s0*s1 x K 
    probs_np = probs_view.numpy()
    cum_probs_np = probs_np.cumsum(axis=1) # s0*s1 x K 
    unif = np.random.uniform(0.0, 1.0, size=[ s0*s1, num ]) # s0*s1 x NUM
    bound = cum_probs_np[:, -1] * unif.T # NUM x s0*s1
    acc = cum_probs_np[np.newaxis,:,:] < bound[:,:,np.newaxis] 
    # NUM x s0*s1 x K 
    acc = np.sum(acc, axis=2).T # s0*s1 x NUM
    rst = torch.from_numpy(acc).long()
    rst = rst.view(s0, s1, num) # s0 x s1 x NUM
    return rst

def sample_uniform(s0, s1, num, event_num): 
    # probs are unnormalized
    probs = torch.zeros(s0, s1, num, device=device).uniform_(0.0, 1.0)
    rst = (probs * event_num).long() # s0 x s1 x NUM
    return rst
