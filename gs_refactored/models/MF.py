import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class MFBPRModel(nn.Module):
    def __init__(self, config, num_user, num_item, device):
        super(MFBPRModel, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.weight_decay = config['weight_decay']
        self.embed_size = config['embed_size']

        self.device = device
        
        self.item_embedding = nn.Embedding(self.num_item, self.embed_size)
        self.user_embedding = nn.Embedding(self.num_user, self.embed_size)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.kaiming_normal_(self.user_embedding.weight)
        nn.init.kaiming_normal_(self.item_embedding.weight)

    def forward(self, user, pos, neg, history, history_mask):
        # user, pos, neg의 dimension은?
        user_embed = self.user_embedding(user)
        pos_embed = self.item_embedding(pos)
        neg_embed = self.item_embedding(neg)
        #print("user_embed:", user_embed)
        #print("pos_embed:", pos_embed)       
        #print("neg_embed:", neg_embed)
        
        # pos_out, neg_out의 dimension은?
        pos_out = torch.mul(user_embed, pos_embed).sum(dim=1)
        neg_out = torch.mul(user_embed, neg_embed).sum(dim=1)
        #print("pos_out:", pos_out)
        #print("neg_out:", neg_out)
        
        out = pos_out - neg_out

        log_prob = F.logsigmoid(out).sum()
        #print("F.logsigmoid(out):", F.logsigmoid(out))
        #print("log_prob:", log_prob)
        reg = self.weight_decay * (user_embed.norm(dim=1).pow(2).sum() + pos_embed.norm(dim=1).pow(2).sum() + neg_embed.norm(dim=1).pow(2).sum())
        #print("reg:", reg)

        return -log_prob + reg, -log_prob, reg, None, user_embed, pos_embed, neg_embed

    @torch.inference_mode()
    def recommend(self, user, history, history_mask, item=None):
        user_embed = F.normalize(self.user_embedding(user))

        if item == None:
            item_embed = self.item_embedding.weight
        else:
            item_embed = self.item_embedding(item)

        user_embed = user_embed.unsqueeze(1)
        # out이 prediction값으로 쓰임
        out = torch.mul(user_embed, item_embed).sum(dim=-1)

        return out, user_embed, item_embed