import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.loss import InfoNCE
from models.layers import EncoderLayer, PositionalEncoding, DIFTransformerLayer

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

        # 1. history_embedding과 history_attention_matrix를 얻어냄
        history_embed, history_att_matrix = self._get_history_embedding(history, history_mask, get_attn_matrix=True)
        
        # 1-1. negative history가 있으면 history embedding을 얻어냄
        neg_history_embed = None
        if neg_history != None:
            neg_history_embed, _ = self._get_history_embedding(neg_history, neg_history_mask)

        # 2. after_history embedding을 얻어냄
        after_history = torch.cat([history[:, 1:], pos.unsqueeze(-1)], dim=-1)
        after_history_mask = torch.cat([history_mask[:, 1:], torch.ones_like(pos).unsqueeze(-1).to(self.device)], dim=-1)

        after_history_embed, _ = self._get_history_embedding(after_history, after_history_mask)
        
        # 3. initial embedding과 after embedding사이 loss 계산
        total_loss, loss_to_track, reg_term = self._get_loss(history_embed, pos_init_embed, neg_init_embed, after_history_embed, pos_last_init_embed, neg_history_embed)
        
        # 3-1. BPR loss일 경우, loss에 regularization term을 붙여줌
        if self.loss_type == "BPR":
            total_loss = total_loss + reg_term
        
        return total_loss, loss_to_track, reg_term, history_att_matrix, history_embed, pos_init_embed, neg_init_embed

    def _get_history_embedding(self, history, history_mask, get_attn_matrix=False):
        # history_embedding에 positional embedding을 추가함
        history_embed = self._add_positional_embedding(self.item_embedding(history))

        # attention mask생성
        attention_mask = self._get_sequence_mask(self.n_heads, history_mask, self.device)
        gaussian_mask = None
        if self.override_mask:
            gaussian_mask = self._get_predefined_attn(self.n_heads, history_mask, self.device)

        history_att_scores = []
        # transformer layer module (SASRec encoder)에 history embedding, attention mask, predefined_attention을 넣어 history_embed를 얻어냄
        for layer in self.transformer_layers:
            history_embed = layer(history_embed, mask=attention_mask, predefined_attn=gaussian_mask)

            if get_attn_matrix:
                history_att_scores.append(layer.scores)

        history_att_matrix = None
        if get_attn_matrix:
            history_att_matrix = torch.mean(torch.cat(history_att_scores, dim=1), dim=1)[:, 0, :]
        history_out = history_embed[:, -1, :]
        
        return history_out, history_att_matrix

    def _get_loss(self, history_embed, pos_item_embed, neg_item_embed, after_history_embed, last_item_embed, neg_history_embed=None):
        loss_func = {"BPR" : self._compute_BPR, "Triplet" : self._compute_triplet, "BCE" : self._compute_BCE, "MBPR": self._compute_MBPR}

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
    
    def _compute_MBPR(self, history_embed, pos_embed, neg_embed, meta):
        pass

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

        # sequence_mask = torch.tril(sequence_mask)
        # sequence_mask를 dim=1에서 n_heads개 만큼 쌓음
        attention_mask = torch.cat([sequence_mask for _ in range(n_heads)], dim=1)

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


class CustomDIFSR(nn.Module):
    def __init__(self, config, num_user, num_item, num_product_type, num_department,
                 num_garment_group, item_product_type_ids, item_department_ids,
                 item_garment_group_ids, device):
        super().__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.weight_decay = config['weight_decay']
        self.embed_size = config['embed_size']
        self.device = device

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.drop_out = config["drop_out"]
        self.seq_len = config["seq_len"]
        self.loss_type = config["loss_type"]
        self.init_scheme = config["init_scheme"]
        self.learnable_pos = config["learnable_pos"]
        self.fusion_type = config.get("fusion_type", "sum")
        self.use_target_projection = config.get("use_target_projection", False)
        self.use_history_meta_projection = config.get("use_history_meta_projection", False)
        self.history_meta_residual_blend = config.get("history_meta_residual_blend", 0.0)
        self.history_meta_scale = config.get("history_meta_scale", 1.0)
        self.target_meta_scale = config.get("target_meta_scale", 1.0)
        self.product_type_scale = config.get("product_type_scale", 1.0)
        self.department_scale = config.get("department_scale", 1.0)
        self.garment_group_scale = config.get("garment_group_scale", 1.0)
        self.active_metadata_features = set(config.get("metadata_features", ["product_type", "department", "garment_group"]))

        self.item_embedding = nn.Embedding(self.num_item, self.embed_size)
        self.product_type_embedding = nn.Embedding(num_product_type, self.embed_size)
        self.department_embedding = nn.Embedding(num_department, self.embed_size)
        self.garment_group_embedding = nn.Embedding(num_garment_group, self.embed_size)
        self.meta_project = nn.Linear(self.embed_size * 3, self.embed_size)
        if self.use_history_meta_projection:
            self.history_meta_project = nn.Linear(self.embed_size * 3, self.embed_size)
        if self.use_target_projection:
            self.target_project = nn.Linear(self.embed_size, self.embed_size)

        self.register_buffer("item_product_type_ids", torch.as_tensor(item_product_type_ids, dtype=torch.long))
        self.register_buffer("item_department_ids", torch.as_tensor(item_department_ids, dtype=torch.long))
        self.register_buffer("item_garment_group_ids", torch.as_tensor(item_garment_group_ids, dtype=torch.long))

        if self.learnable_pos:
            self.positional_encoding = nn.Parameter(torch.empty(self.seq_len, self.embed_size))
            nn.init.kaiming_normal_(self.positional_encoding)
        else:
            self.positional_encoding = PositionalEncoding(hidden_dim=self.embed_size, device=self.device)

        self.transformer_layers = nn.ModuleList([
            DIFTransformerLayer(
                self.n_heads,
                self.embed_size,
                [self.embed_size, self.embed_size, self.embed_size],
                3,
                self.embed_size * 2,
                self.drop_out,
                self.drop_out,
                self.seq_len,
                fusion_type=self.fusion_type,
            )
            for _ in range(self.n_layers)
        ])

        if self.loss_type == "BCE":
            self.activation = nn.Sigmoid()

        self._init_weight()

    def _is_feature_enabled(self, feature_name):
        return feature_name in self.active_metadata_features

    def _mask_feature_embedding(self, feature_name, embed):
        if self._is_feature_enabled(feature_name):
            scale = {
                "product_type": self.product_type_scale,
                "department": self.department_scale,
                "garment_group": self.garment_group_scale,
            }.get(feature_name, 1.0)
            return embed * scale
        return torch.zeros_like(embed)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                if self.init_scheme in ["Kaiming", "kaiming"]:
                    torch.nn.init.kaiming_normal_(m.weight)
                elif self.init_scheme in ["Xavier", "xavier"]:
                    torch.nn.init.xavier_normal_(m.weight)

    def _lookup_item_attributes(self, item_ids):
        product_type = self._mask_feature_embedding(
            "product_type",
            self.product_type_embedding(self.item_product_type_ids[item_ids]),
        ).unsqueeze(-2)
        department = self._mask_feature_embedding(
            "department",
            self.department_embedding(self.item_department_ids[item_ids]),
        ).unsqueeze(-2)
        garment_group = self._mask_feature_embedding(
            "garment_group",
            self.garment_group_embedding(self.item_garment_group_ids[item_ids]),
        ).unsqueeze(-2)
        return [product_type, department, garment_group]

    def _get_item_representation(self, item_ids, meta_scale=1.0, history_mode=False):
        item_embed = self.item_embedding(item_ids)
        product_type_embed = self._mask_feature_embedding(
            "product_type",
            self.product_type_embedding(self.item_product_type_ids[item_ids]),
        )
        department_embed = self._mask_feature_embedding(
            "department",
            self.department_embedding(self.item_department_ids[item_ids]),
        )
        garment_group_embed = self._mask_feature_embedding(
            "garment_group",
            self.garment_group_embedding(self.item_garment_group_ids[item_ids]),
        )
        meta_embed = torch.cat([product_type_embed, department_embed, garment_group_embed], dim=-1)
        shared_meta_embed = self.meta_project(meta_embed)
        if history_mode and self.use_history_meta_projection:
            history_meta_embed = self.history_meta_project(meta_embed)
            meta_embed = shared_meta_embed + (self.history_meta_residual_blend * history_meta_embed)
        else:
            meta_embed = shared_meta_embed
        return item_embed + (meta_scale * meta_embed)

    def _get_target_representation(self, item_ids):
        item_repr = self._get_item_representation(item_ids, meta_scale=self.target_meta_scale)
        if self.use_target_projection:
            item_repr = self.target_project(item_repr)
        return item_repr

    def _get_position_embedding(self, batch_size, seq_len):
        if self.learnable_pos:
            return self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        return self.positional_encoding.encoding[:, :seq_len, :].expand(batch_size, -1, -1)

    def _get_attention_mask(self, history_mask):
        seq_len = history_mask.shape[1]
        mask_1 = history_mask.unsqueeze(1).repeat(1, seq_len, 1)
        mask_2 = history_mask.unsqueeze(2).repeat(1, 1, seq_len)
        sequence_mask = (mask_1 * mask_2).unsqueeze(1).float().to(self.device)
        return (1.0 - sequence_mask) * -10000.0

    def _get_history_embedding(self, history, history_mask):
        batch_size, seq_len = history.shape
        history_embed = self._get_item_representation(history, meta_scale=self.history_meta_scale, history_mode=True)
        position_embedding = self._get_position_embedding(batch_size, seq_len)
        attribute_embed = self._lookup_item_attributes(history)
        attention_mask = self._get_attention_mask(history_mask)

        for layer in self.transformer_layers:
            history_embed = layer(history_embed, attribute_embed, position_embedding, attention_mask)

        return history_embed[:, -1, :]

    def _compute_BPR(self, history_embed, pos_embed, neg_embed):
        pos_preds = torch.mul(history_embed, pos_embed).sum(dim=-1, keepdims=True)
        neg_preds = torch.mul(history_embed, neg_embed).sum(dim=-1, keepdims=True)
        return -F.logsigmoid(pos_preds - neg_preds).sum()

    def _compute_triplet(self, history_embed, pos_embed, neg_embed):
        criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
        )
        return criterion(
            F.normalize(history_embed, dim=-1),
            F.normalize(pos_embed, dim=-1),
            F.normalize(neg_embed, dim=-1),
        )

    def _compute_BCE(self, history_embed, pos_embed, neg_embed):
        pos_preds = self.activation(torch.mul(history_embed, pos_embed).sum(dim=-1, keepdims=True))
        neg_preds = self.activation(torch.mul(history_embed, neg_embed).sum(dim=-1, keepdims=True))
        preds = torch.cat([pos_preds, neg_preds], dim=-1)
        labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)], dim=-1).to(self.device)
        return nn.BCELoss()(preds, labels)

    def _get_loss(self, history_embed, pos_item_embed, neg_item_embed):
        loss_func = {"BPR": self._compute_BPR, "Triplet": self._compute_triplet, "BCE": self._compute_BCE}
        history_anchor_loss = loss_func[self.loss_type](history_embed, pos_item_embed, neg_item_embed)
        raw_reg_term = (
            history_embed.norm(dim=1).pow(2).sum()
            + pos_item_embed.norm(dim=1).pow(2).sum()
            + neg_item_embed.norm(dim=1).pow(2).sum()
        )
        reg_term = self.weight_decay * raw_reg_term
        return history_anchor_loss, history_anchor_loss, reg_term

    def forward(self, user, pos, prodtype, depart, garmgroup, age, neg, history, history_mask,
                neg_history=None, neg_history_mask=None):
        del user, prodtype, depart, garmgroup, age, neg_history, neg_history_mask
        pos_init_embed = self._get_target_representation(pos)
        neg_init_embed = self._get_target_representation(neg)
        history_embed = self._get_history_embedding(history, history_mask)
        total_loss, loss_to_track, reg_term = self._get_loss(history_embed, pos_init_embed, neg_init_embed)
        if self.loss_type == "BPR":
            total_loss = total_loss + reg_term
        return total_loss, loss_to_track, reg_term, None, history_embed, pos_init_embed, neg_init_embed

    @torch.inference_mode()
    def recommend(self, user, history, history_mask, item=None):
        del user
        history_embed = self._get_history_embedding(history, history_mask).unsqueeze(1)

        if item is None:
            all_items = torch.arange(self.num_item, device=self.device, dtype=torch.long)
            item_init_embed = self._get_target_representation(all_items)
        else:
            item_init_embed = self._get_target_representation(item)

        if self.loss_type == "BCE":
            out = torch.mul(history_embed, item_init_embed).sum(dim=-1)
        else:
            history_embed = F.normalize(history_embed, dim=-1)
            item_init_embed = F.normalize(item_init_embed, dim=-1)
            out = torch.mul(history_embed, item_init_embed).sum(dim=-1)

        return out, history_embed, item_init_embed
    
class CustomMetaSASRec(nn.Module):
    def __init__(self, config, num_user, num_item, num_product_code, num_product_type, num_graphical_appearance, 
                 num_colour_group, num_perceived_colour_value, num_perceived_colour_master, num_department,
                 num_index_group, num_section, num_garment_group, num_age, num_price, device):
        super().__init__()
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
        self.num_age = num_age
        self.num_price = num_price
        
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
        #self.product_code_embedding = nn.Embedding(self.num_product_code, self.embed_size)
        self.product_type_embedding = nn.Embedding(self.num_product_type, self.embed_size)
        #self.graphical_appearance_embedding = nn.Embedding(self.num_graphical_appearance, self.embed_size)
        #self.colour_group_embedding = nn.Embedding(self.num_colour_group, self.embed_size)
        #self.perceived_colour_value = nn.Embedding(self.num_perceived_colour_value, self.embed_size)
        #self.perceived_colour_master = nn.Embedding(self.num_perceived_colour_master, self.embed_size)
        self.department = nn.Embedding(self.num_department, self.embed_size)
        #self.index_group = nn.Embedding(self.num_index_group, self.embed_size)
        #self.section = nn.Embedding(self.num_section, self.embed_size)
        self.garment_group = nn.Embedding(self.num_garment_group, self.embed_size)
        self.age = nn.Embedding(self.num_age, self.embed_size)
        #self.price = nn.Embedding(self.num_price, self.embed_size)
        
        self.project = nn.Linear(4*self.embed_size, self.embed_size)

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

    def forward(self, user, pos, prodtype, depart, garmgroup, age,
                neg, history, history_mask, neg_history=None, neg_history_mask=None):
        pos_init_embed, neg_init_embed = self.item_embedding(pos), self.item_embedding(neg)
        pos_last_init_embed = self.item_embedding(history[:,-1])

        #prodcode_embed = self.product_code_embedding(prodcode)
        prodtype_embed = self.product_type_embedding(prodtype)
        #graph_appear_embed = self.graphical_appearance_embedding(graph_appear)
        #colour_group_embed = self.colour_group_embedding(colour_group)
        #perceived_colour_value_embed = self.perceived_colour_value(pcolval)
        #perceived_colour_master_embed = self.perceived_colour_master(pcolmas)
        department_embed = self.department(depart)
        #index_group_embed = self.index_group(idxgroup)
        #section_embed = self.section(section)
        garment_group_embed = self.garment_group(garmgroup)
        age_embed = self.age(age)
        #price_embed = self.price(price)
        
        # 1. average_all
        # meta_embed = (prodcode_embed + prodtype_embed + graph_appear_embed + colour_group_embed + perceived_colour_value_embed 
        #              + perceived_colour_master_embed + department_embed + index_group_embed + section_embed + garment_group_embed)/10
        
        # 2. add_all
        # meta_embed = prodcode_embed + prodtype_embed + graph_appear_embed + colour_group_embed + perceived_colour_value_embed 
        #              + perceived_colour_master_embed + department_embed + index_group_embed + section_embed + garment_group_embed
        
        # 3. average_partial (neglect personal preference)
        # meta_embed = (prodcode_embed + prodtype_embed + department_embed + index_group_embed + section_embed + garment_group_embed)/6
        
        # 4. add_partial (neglect personal preference)
        # meta_embed = (prodcode_embed + prodtype_embed + department_embed + index_group_embed + section_embed + garment_group_embed)/6
        
        # 5. 8 feature concat (neglect perceived_colour_value, perceived_colour_master)
        #meta_embed = torch.cat([prodcode_embed, prodtype_embed, graph_appear_embed, colour_group_embed, department_embed, index_group_embed, section_embed, garment_group_embed], dim=-1)
        
        # 6-1. important feature divide & concat
        #meta_embed = torch.cat([prodtype_embed, department_embed, garment_group_embed, age_embed], dim=-1)
        
        # 6-2. important feature addition
        #meta_embed = prodtype_embed + department_embed + garment_group_embed + age_embed

        # 6-3. important feature average
        #meta_embed = (prodtype_embed + department_embed + garment_group_embed + age_embed)/4
        
        # 6-4. important feature concat & project
        meta_embed = torch.cat([prodtype_embed, department_embed, garment_group_embed, age_embed], dim=-1)
        meta_embed = self.project(meta_embed)

        #pos_init_embed = torch.cat([pos_init_embed, prodtype_embed, department_embed, garment_group_embed], dim=-1)
        #neg_init_embed = torch.cat([neg_init_embed, prodtype_embed, department_embed, garment_group_embed], dim=-1)
        #pos_last_init_embed = torch.cat([pos_last_init_embed, prodtype_embed, department_embed, garment_group_embed], dim=-1)
        
        # 7. all concat and project to embed_size
        #meta_embed = torch.cat([prodcode_embed, prodtype_embed, graph_appear_embed, colour_group_embed, perceived_colour_value_embed, 
        #                        perceived_colour_master_embed, department_embed, index_group_embed, section_embed, garment_group_embed], dim=-1)
        #meta_embed = self.project(meta_embed)
        
        pos_init_embed = pos_init_embed + meta_embed
        neg_init_embed = neg_init_embed + meta_embed
        pos_last_init_embed = pos_last_init_embed + meta_embed
        
        #meta_embed = torch.as_tensor(meta_embed, dtype=torch.long)
        #meta_embed = self.meta_embedding(meta_embed)
 
        # 1. history_embedding과 history_attention_matrix를 얻어냄
        history_embed, history_att_matrix = self._get_history_embedding(history, history_mask, get_attn_matrix=True)
        
        # 1-1. negative history가 있으면 history embedding을 얻어냄
        neg_history_embed = None
        if neg_history != None:
            neg_history_embed, _ = self._get_history_embedding(neg_history, neg_history_mask)

        # 2. after_history embedding을 얻어냄
        after_history = torch.cat([history[:, 1:], pos.unsqueeze(-1)], dim=-1)
        after_history_mask = torch.cat([history_mask[:, 1:], torch.ones_like(pos).unsqueeze(-1).to(self.device)], dim=-1)

        after_history_embed, _ = self._get_history_embedding(after_history, after_history_mask)
        
        # 3. initial embedding과 after embedding사이 loss 계산
        total_loss, loss_to_track, reg_term = self._get_loss(history_embed, pos_init_embed, neg_init_embed, after_history_embed, pos_last_init_embed, neg_history_embed)
        
        # 3-1. BPR loss일 경우, loss에 regularization term을 붙여줌
        if self.loss_type == "BPR":
            total_loss = total_loss + reg_term
        
        return total_loss, loss_to_track, reg_term, history_att_matrix, history_embed, pos_init_embed, neg_init_embed

    def _get_history_embedding(self, history, history_mask, get_attn_matrix=False):
        # history_embedding에 positional embedding을 추가함
        history_embed = self._add_positional_embedding(self.item_embedding(history))

        # attention mask생성
        attention_mask = self._get_sequence_mask(self.n_heads, history_mask, self.device)
        gaussian_mask = None
        if self.override_mask:
            gaussian_mask = self._get_predefined_attn(self.n_heads, history_mask, self.device)

        history_att_scores = []
        # transformer layer module (SASRec encoder)에 history embedding, attention mask, predefined_attention을 넣어 history_embed를 얻어냄
        for layer in self.transformer_layers:
            history_embed = layer(history_embed, mask=attention_mask, predefined_attn=gaussian_mask)

            if get_attn_matrix:
                history_att_scores.append(layer.scores)

        history_att_matrix = None
        if get_attn_matrix:
            history_att_matrix = torch.mean(torch.cat(history_att_scores, dim=1), dim=1)[:, 0, :]
        history_out = history_embed[:, -1, :]
        
        return history_out, history_att_matrix

    def _get_loss(self, history_embed, pos_item_embed, neg_item_embed, after_history_embed, last_item_embed, neg_history_embed=None):
        loss_func = {"BPR" : self._compute_BPR, "Triplet" : self._compute_triplet, "BCE" : self._compute_BCE, "MBPR": self._compute_MBPR}

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
    
    def _compute_MBPR(self, history_embed, pos_embed, neg_embed, meta):
        pass

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

        # sequence_mask = torch.tril(sequence_mask)
        # sequence_mask를 dim=1에서 n_heads개 만큼 쌓음
        attention_mask = torch.cat([sequence_mask for _ in range(n_heads)], dim=1)
        
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
