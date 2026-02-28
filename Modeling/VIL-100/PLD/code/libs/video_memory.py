import numpy as np
from libs.utils import *

class Video_Memory(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def forward_for_dict_initialization(self):
        self.keylist = ['img_feat', 'prob_map', 'coeff_map', 'guide_cls']
        self.data = dict()
        for key in self.keylist:
            self.data[key] = dict()
        self.memory_t = 0

    def forward_for_dict_memorization(self):
        for i in range(self.memory_t - 1, -1, -1):
            for key in self.keylist:
                self.data[key][f't-{i+1}'] = self.data[key][f't-{i}']#倒推时间步上的数据 为下一帧腾出空间

        for key in self.keylist:
            self.data[key].pop('t-0')#删除最新的时间步的数据 因为t-0已经转移到t-1了
        if self.memory_t >= self.cfg.num_t:
            self.memory_t -= 1

    def forward_for_dict_initialization_per_frame(self, t):
        for key in self.keylist:
            self.data[key][t] = dict()
        self.t = t

    def forward_for_dict_update_per_frame(self, model, batch_idx=None, mode=None):
        if mode == 'intra' and batch_idx is not None:
            self.data['img_feat'][self.t] = model.img_feat[batch_idx:batch_idx + 1]#三个维度的特征图的合并
            self.data['prob_map'][self.t] = model.prob_map[batch_idx:batch_idx + 1]
            self.memory_t += 1
        elif mode == 'intra' and batch_idx is None:
            self.data['img_feat'][self.t] = model.img_feat
            self.data['prob_map'][self.t] = model.prob_map
            self.memory_t += 1
        elif mode == 'update':
            self.data['img_feat'][self.t] = model.img_feat.detach()
            self.data['prob_map'][self.t] = model.prob_map.detach()

    def forward_for_dict_transfer(self, model):
        model.memory = dict()
        for key in self.keylist:
            model.memory[key] = self.data[key]

        return model
