import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.loss import InfoNCE
from models.layers import EncoderLayer, PositionalEncoding

class CustomSASRec(nn.Module):
    def __init__(self, config, num_user, num_item, device):
        super().__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.weight_decay = config['weight_decay']
        self.embed_size = config['embed_size']
        self.use_pretrain = False

        self.device = device

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.drop_out = config["drop_out"]
        self.seq_len = config["seq_len"]

        self.loss_type = config["loss_type"]
        self.init_scheme = config["init_scheme"]
        self.sampler_type = config["sampler_type"]

        self.learnable_pos = config["learnable_pos"]
        self.override_mask = config["override_mask"]

        self.item_embedding = nn.Embedding(self.num_item, self.embed_size)

        if config["learnable_pos"]:
            self.positional_encoding = nn.Parameter(torch.empty(self.seq_len, self.embed_size))
            nn.init.kaiming_normal_(self.positional_encoding)
        else:
            self.positional_encoding = PositionalEncoding(hidden_dim=self.embed_size, device=self.device)

        self.transformer_layers = nn.ModuleList([EncoderLayer(self.embed_size, self.n_heads, self.drop_out) for _ in range(self.n_layers)])

        if config["loss_type"] == "BCE":
            self.activation = nn.Sigmoid()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                if self.init_scheme in ["Kaiming", "kaiming"]: 
                    torch.nn.init.kaiming_normal_(m.weight)
                elif self.init_scheme in ["Xavier", "xavier"]:
                    torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.ReLU):
                m = nn.ReLU(inplace=False)

    def forward(self, user, pos, neg, history, history_mask, neg_history=None, neg_history_mask=None):
        pos_init_embed, neg_init_embed = self.item_embedding(pos), self.item_embedding(neg)
        pos_last_init_embed = self.item_embedding(history[:,-1])
        #print("pos.shape:", pos.shape, "neg.shape:", neg.shape)
        #print("pos_init_embed.shape:", pos_init_embed.shape, "neg_init_embed.shape:", neg_init_embed.shape)
        #print("pos_last_init_embed.shape:", pos_last_init_embed.shape)

        # 1. history_embedding과 history_attention_matrix를 얻어냄
        history_embed, history_att_matrix = self._get_history_embedding(history, history_mask, get_attn_matrix=True)
        #print("history.shape:", history.shape)
        #print("history_mask.shape:", history_mask.shape)
        #print("history_embed.shape:", history_embed.shape)
        
        # 1-1. negative history가 있으면 history embedding을 얻어냄
        neg_history_embed = None
        if neg_history != None:
            neg_history_embed, _ = self._get_history_embedding(neg_history, neg_history_mask)
            #print("neg_history_embed.shape:", neg_history_embed.shape)

        # 2. after_history embedding을 얻어냄
        after_history = torch.cat([history[:, 1:], pos.unsqueeze(-1)], dim=-1)
        after_history_mask = torch.cat([history_mask[:, 1:], torch.ones_like(pos).unsqueeze(-1).to(self.device)], dim=-1)
        #print("after_history.shape:", after_history.shape, "after_history_mask.shape:", after_history_mask.shape)

        after_history_embed, _ = self._get_history_embedding(after_history, after_history_mask)
        #print("after_history_embed.shape:", after_history_embed.shape)
        
        # 3. initial embedding과 after embedding사이 loss 계산
        total_loss, loss_to_track, reg_term = self._get_loss(history_embed, pos_init_embed, neg_init_embed, after_history_embed, pos_last_init_embed, neg_history_embed)
        
        # 3-1. BPR loss일 경우, loss에 regularization term을 붙여줌
        if self.loss_type == "BPR":
            total_loss = total_loss + reg_term
        
        return total_loss, loss_to_track, reg_term, history_att_matrix, history_embed, pos_init_embed, neg_init_embed

    def _get_history_embedding(self, history, history_mask, get_attn_matrix=False):
        # history_embedding에 positional embedding을 추가함
        history_embed = self._add_positional_embedding(self.item_embedding(history))
        #print("history.shape:", history.shape) # (B, N)
        #print("self.item_embedding(history).shape:", self.item_embedding(history).shape) # (B, N, D)
        #print("after adding positional embedding:", history_embed.shape) # (B, N, D)

        # attention mask생성
        attention_mask = self._get_sequence_mask(self.n_heads, history_mask, self.device)
        #print("attention_mask.shape:", attention_mask.shape)
        gaussian_mask = None
        if self.override_mask:
            gaussian_mask = self._get_predefined_attn(self.n_heads, history_mask, self.device)
            #print("gaussian_mask.shape:", gaussian_mask.shape)

        history_att_scores = []
        # transformer layer module (SASRec encoder)에 history embedding, attention mask, predefined_attention을 넣어 history_embed를 얻어냄
        for layer in self.transformer_layers:
            history_embed = layer(history_embed, mask=attention_mask, predefined_attn=gaussian_mask)

            if get_attn_matrix:
                history_att_scores.append(layer.scores)
        #print("history_embed after passing layer:", history_embed.shape)

        history_att_matrix = None
        if get_attn_matrix:
            history_att_matrix = torch.mean(torch.cat(history_att_scores, dim=1), dim=1)[:, 0, :]
        #print("history_att_matrix.shape:", history_att_matrix.shape)
        history_out = history_embed[:, -1, :]
        #print("history_out.shape:", history.shape)
        
        return history_out, history_att_matrix

    def _get_loss(self, history_embed, pos_item_embed, neg_item_embed, after_history_embed, last_item_embed, neg_history_embed=None):
        loss_func = {"BPR" : self._compute_BPR, "Triplet" : self._compute_triplet, "BCE" : self._compute_BCE}

        history_anchor_loss = loss_func[self.loss_type](history_embed, pos_item_embed, neg_item_embed)
        raw_reg_term = (history_embed.norm(dim=1).pow(2).sum() + pos_item_embed.norm(dim=1).pow(2).sum() + neg_item_embed.norm(dim=1).pow(2).sum())

        total_loss = history_anchor_loss
        reg_term = self.weight_decay * raw_reg_term

        return total_loss, history_anchor_loss, reg_term

    @torch.inference_mode()
    def recommend(self, user, history, history_mask, item=None):
        history_embed, _ = self._get_history_embedding(history, history_mask)
        history_embed = history_embed.unsqueeze(1)

        if item == None:
            item_init_embed = self.item_embedding.weight
        else:
            item_init_embed = self.item_embedding(item)

        if self.loss_type == "BCE":
            # out = self.activation(torch.mul(history_out, item_init_embed).sum(dim=-1))
            out = torch.mul(history_embed, item_init_embed).sum(dim=-1)
        else:
            history_embed, item_init_embed = F.normalize(history_embed, dim=-1), F.normalize(item_init_embed, dim=-1)
            out = torch.mul(history_embed, item_init_embed).sum(dim=-1)

        return out, history_embed, item_init_embed

    def _add_positional_embedding(self, history_embed):
        if self.learnable_pos:
            history_embed = history_embed + torch.tile(self.positional_encoding.unsqueeze(0), (history_embed.shape[0], 1, 1))
        else:
            history_embed = self.positional_encoding(history_embed)

        return history_embed

    def _compute_BPR(self, history_embed, pos_embed, neg_embed):
        pos_preds = torch.mul(history_embed, pos_embed).sum(dim=-1, keepdims=True)
        neg_preds = torch.mul(history_embed, neg_embed).sum(dim=-1, keepdims=True)

        out = pos_preds - neg_preds

        log_prob = F.logsigmoid(out).sum()
        loss = -log_prob

        return loss

    def _compute_triplet(self, history_embed, pos_embed, neg_embed):
        criterion = nn.TripletMarginWithDistanceLoss(distance_function = lambda x, y : 1.0 - F.cosine_similarity(x, y))
        loss = criterion(F.normalize(history_embed, dim=-1), F.normalize(pos_embed, dim=-1), F.normalize(neg_embed, dim=-1))

        return loss

    def _compute_BCE(self, history_embed, pos_embed, neg_embed):
        pos_preds = self.activation(torch.mul(history_embed, pos_embed).sum(dim=-1, keepdims=True))
        if self.sampler_type == "MultipleNegative":
            neg_preds = self.activation(torch.mul(history_embed.unsqueeze(dim=1), neg_embed).sum(dim=-1))
        else:
            neg_preds = self.activation(torch.mul(history_embed, neg_embed).sum(dim=-1, keepdims=True))

        preds, labels = torch.cat([pos_preds, neg_preds], dim=-1), torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)], dim=-1).to(self.device)

        criterion = nn.BCELoss()
        loss = criterion(preds, labels)

        return loss

    def _get_sequence_mask(self, n_heads, mask, device):
        batch_size, sequence_len = mask.shape[:2]
        
        mask_1 = mask.unsqueeze(1).repeat(1, sequence_len, 1)
        mask_2 = mask.unsqueeze(2).repeat(1, 1, sequence_len)
        sequence_mask = (mask_1 * mask_2).unsqueeze(1).to(device)
        #print("mask_1:", mask_1)
        #print("mask_2:", mask_2)
        #print("sequence_mask:", sequence_mask)
        #print("mask_1.shape:", mask_1.shape, "mask_2.shape:", mask_2.shape, "sequence_mask.shape:", sequence_mask.shape)

        # sequence_mask = torch.tril(sequence_mask)
        # sequence_mask를 dim=1에서 n_heads개 만큼 쌓음
        attention_mask = torch.cat([sequence_mask for _ in range(n_heads)], dim=1)
        #print("attention_mask.shape:", attention_mask.shape)
        # tensor 연산 추적을 끔 (True일 경우 추적) -> memory를 아끼기 위해?
        attention_mask.requires_grad = False

        return attention_mask

    def _get_predefined_attn(self, n_heads, mask, device):
        batch_size, sequence_len = mask.shape[:2]
        
        mask_1 = mask.unsqueeze(1).repeat(1, sequence_len, 1)
        mask_2 = mask.unsqueeze(2).repeat(1, 1, sequence_len)
        sequence_mask = (mask_1 * mask_2).unsqueeze(1).to(device)
        
        gaussian_filter = torch.cat([torch.exp(-1/2 * (torch.arange(sequence_len) - float(idx))**2 / (float(sequence_len//10)**2)).unsqueeze(0) for idx in range(sequence_len)], dim=0)
        gaussian_filter = gaussian_filter.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        gaussian_mask = gaussian_filter.unsqueeze(1) * sequence_mask
        gaussian_mask = (gaussian_mask / (torch.sum(gaussian_mask, dim=-1, keepdims=True) + 1e-8)).to(self.device)

        attention_mask = torch.cat([gaussian_mask for _ in range(n_heads)], dim=1)

        attention_mask.requires_grad = False

        return attention_mask