import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class MFBPRModel(nn.Module):
    def __init__(self, config, num_user, num_item, num_product_code, num_product_type, num_graphical_appearance, 
                 num_colour_group, num_perceived_colour_value, num_perceived_colour_master, num_department,
                 num_index_group, num_section, num_garment_group, device):
        super(MFBPRModel, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_product_code = num_product_code
        self.num_product_type = num_product_type
        self.num_graphical_appearance = num_graphical_appearance
        self.num_colour_group = num_colour_group
        self.num_perceived_colour_value = num_perceived_colour_value
        self.num_perceived_colour_master = num_perceived_colour_master
        self.num_department = num_department
        self.num_index_group = num_index_group
        self.num_section = num_section
        self.num_garment_group = num_garment_group
        
        self.weight_decay = config['weight_decay']
        self.embed_size = config['embed_size']

        self.device = device
        
        self.item_embedding = nn.Embedding(self.num_item, self.embed_size)
        self.product_code_embedding = nn.Embedding(self.num_product_code, self.embed_size)
        self.product_type_embedding = nn.Embedding(self.num_product_type, self.embed_size)
        self.graphical_appearance_embedding = nn.Embedding(self.num_graphical_appearance, self.embed_size)
        self.colour_group_embedding = nn.Embedding(self.num_colour_group, self.embed_size)
        self.perceived_colour_value = nn.Embedding(self.num_perceived_colour_value, self.embed_size)
        self.perceived_colour_master = nn.Embedding(self.num_perceived_colour_master, self.embed_size)
        self.department = nn.Embedding(self.num_department, self.embed_size)
        self.index_group = nn.Embedding(self.num_index_group, self.embed_size)
        self.section = nn.Embedding(self.num_section, self.embed_size)
        self.garment_group = nn.Embedding(self.num_garment_group, self.embed_size)

        self.user_embedding = nn.Embedding(self.num_user, self.embed_size)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.kaiming_normal_(self.user_embedding.weight)
        nn.init.kaiming_normal_(self.item_embedding.weight)

    def forward(self, user, pos, prodcode, prodtype, graph_appear, colour_group, 
                pcolval, pcolmas, depart, idxgroup, section, 
                garmgroup, neg, history, history_mask):
        # user, pos, neg의 dimension은?
        user_embed = self.user_embedding(user)
        pos_embed = self.item_embedding(pos)
        prodcode_embed = self.product_code_embedding(prodcode)
        prodtype_embed = self.product_type_embedding(prodtype)
        graph_appear_embed = self.graphical_appearance_embedding(graph_appear)
        colour_group_embed = self.colour_group_embedding(colour_group)
        perceived_colour_value_embed = self.perceived_colour_value(pcolval)
        perceived_colour_master_embed = self.perceived_colour_master(pcolmas)
        department_embed = self.department(depart)
        index_group_embed = self.index_group(idxgroup)
        section_embed = self.section(section)
        garment_group_embed = self.garment_group(garmgroup)        
        
        neg_embed = self.item_embedding(neg)
        #print("user_embed:", user_embed)
        #print("pos_embed:", pos_embed)       
        #print("neg_embed:", neg_embed)
        meta_embed = torch.cat([prodcode_embed, prodtype_embed, graph_appear_embed, colour_group_embed, 
                                perceived_colour_value_embed, perceived_colour_master_embed, department_embed, 
                                index_group_embed, section_embed, garment_group_embed], dim=1)
        
        # pos_out, neg_out의 dimension은?
        pos_embed = pos_embed + meta_embed
        neg_embed = neg_embed + meta_embed
        
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