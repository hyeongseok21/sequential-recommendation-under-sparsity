import os
import sys
import json
import time
import uuid
import pickle
import argparse
import collections
from tqdm import tqdm
from collections import defaultdict

import mlflow
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# Dataset: 샘플과 정답을 저장
# DataLoader: 샘플에 쉽게 접근할 수 있도록 iterable한 객체로 감쌈

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from hm_preprocess import hm_prep
from hm_preprocess_meta import hm_prep_meta
from dataset import TrainDataset, TestDataset, BenchmarkDataset, BenchmarkTotalDataset
from dataset import NegativeSampler, TwoViewSampler, TwoViewConsistentSampler, TwoViewRandomSampler
from util.helper import init_logger
from util.metric import hit, ndcg, map_, diversity

from models.scheduler import get_scheduler
import matplotlib.pyplot as plt
import matplotlib

from visualizer import dosnes
from mpl_toolkits.mplot3d import Axes3D

class Trainer:
    def __init__(self, config_path, config=None):
        self.logger = init_logger('trainer')
        
        # 0. config file을 불러와서 parameter 초기화
        if config is None:
            with open(config_path, 'rb') as f:
                config = json.load(f)
        self.config = config

        self.logger.info('Parameter settings : {}'.format(json.dumps(self.config, indent=4)))
        self.dataset_params = self.config['dataset_params']
        self.model_params = self.config['model_params']
        self.train_params = self.config['train_params']
        self.eval_params = self.config['eval_params']

        self.logger.info('Loading Dataset.')
        dataset_path = os.path.join(self.dataset_params['dataset_path'], self.dataset_params['save_name'] + '.pkl')
        if not os.path.isfile(dataset_path): 
            self.logger.info('No preprocessed data')  
            self.data_dict = hm_prep(self.dataset_params)
        elif self.dataset_params['embed_metadata']:
            self.logger.info('metadata preprocess starts')
            self.data_dict = hm_prep_meta(self.dataset_params)
        else:
            self.logger.info('Preprocessed data already exist. Loading...')  
            with open(dataset_path, 'rb') as f:
                self.data_dict = pickle.load(f)
        self.logger.info('Number of users : {}\tNumber of items : {}'.format(
            self.data_dict['num_user'], self.data_dict['num_item'])
        )

        # 1. train, test, benchmark dataset 초기화
        self.logger.info('Initialize Dataset & Negative Sampler.')
        train_ds = TrainDataset(self.data_dict, self.model_params['seq_len'])
        test_ds = TestDataset(self.data_dict, self.model_params['seq_len'])
        benchmark_ds = BenchmarkTotalDataset(self.data_dict, self.model_params['seq_len'], num_negatives=100)
        
        # 2. sampler_type에 따라 Sampler 초기화
        if self.model_params['sampler_type'] == "Negative":
            sampler = NegativeSampler(self.data_dict)
        elif self.model_params['sampler_type'] == "TwoView":
            sampler = TwoViewSampler(self.data_dict, self.model_params['seq_len'])
        
        # 3. train, test, benchmark DataLoader 정의
        self.train_dl = DataLoader(
            train_ds,
            collate_fn=sampler.sampling,
            batch_size=self.train_params['batch_size'],
            shuffle=True,
            num_workers=8
        )
        if self.eval_params['test']:
            self.test_dl = DataLoader(
                test_ds,
                batch_size=self.eval_params['batch_size_test'],
                shuffle=False,
                num_workers=8
            )
        if self.eval_params['benchmark']:
            self.benchmark_dl = DataLoader(
                benchmark_ds,
                batch_size=self.eval_params['batch_size_benchmark'],
                shuffle=False,
                num_workers=8
            )

        self.logger.info('Initializing Model.')
        self.device = torch.device('cuda:{}'.format(self.train_params['device_num']))
        
        # 4. model_type에 따라 self.model 초기화
        if self.model_params['model_type'] == "MF":
            from models.MF import MFBPRModel
            get_model = MFBPRModel
        elif self.model_params['model_type'] == "Transformer":
            from models.Transformer import CustomSASRec
            get_model = CustomSASRec

        self.model = get_model(
            self.model_params, 
            num_user = self.data_dict['num_user'],
            num_item = self.data_dict['num_item'],
            num_product_code = self.data_dict['num_product_code'],
            num_product_type = self.data_dict['num_product_type'],
            num_graphical_appearance = self.data_dict['num_graphical_appearance'],
            num_colour_group = self.data_dict['num_colour_group'],
            num_perceived_colour_value = self.data_dict['num_perceived_colour_value'],
            num_perceived_colour_master = self.data_dict['num_perceived_colour_master'],
            num_department = self.data_dict['num_department'],
            num_index_group = self.data_dict['num_index_group'],
            num_section = self.data_dict['num_section'],
            num_garment_group = self.data_dict['num_garment_group'],
            device=self.device
        ).to(self.device)

        # 4-1. weight_decay option 활성화 시, optimizer에 weight_decay를 설정
        if self.model_params['weight_decay_opt']:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.train_params['lr'], weight_decay=self.model_params['weight_decay'])
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.train_params['lr'])

        #total_steps = int(len(self.train_dl) / self.train_params['batch_size']) * self.train_params['train_epoch']
        #warmup_steps = total_steps // 10

        #if self.train_params['scheduler_type'] != 'original':
        #    self.scheduler = get_scheduler(self.opt, self.train_params['scheduler_type'], total_steps, warmup_steps)

        # 5. popular_ten flag가 켜져있으면 self.popular_items를 계산하고 저장
        if self.eval_params['popular_ten']:
            item_count = collections.Counter(self.data_dict['train_df']['item_id'].to_numpy())
            
            def invert_dict(d):
                result = collections.defaultdict(set)
                for k in d:
                    result[d[k]].add(k)
                return dict(result)
            
            item_count_inv = invert_dict(item_count)
            popular_items_count = sorted(list(item_count_inv.keys()), reverse=True)[:self.train_params['top_k']]
            self.popular_items = []
            for per_item_count in popular_items_count:
                if len(self.popular_items) > self.train_params['top_k']:
                    break
                self.popular_items = self.popular_items + list(item_count_inv[per_item_count])[:(self.train_params['top_k'] - len(self.popular_items))]

    # config 저장
    def save_config(self):
        save_dir = os.path.join(self.train_params['save_path'], self.train_params['save_name'])
        os.makedirs(save_dir, exist_ok=True)
        result_path = os.path.join(save_dir, 'config.json')
        with open(result_path, 'w') as f:
            json.dump(dict(self.config), f, indent=4)

    # checkpoint 저장
    def save_checkpoint(self, epoch, best_hr, best_ndcg, best_epoch, model):
        save_dir = os.path.join(self.train_params['save_path'], self.train_params['save_name'])
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, '{:06d}_epoch.pth'.format(epoch))
        torch.save({
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "best_hr": best_hr,
            "best_ndcg": best_ndcg,
            "best_epoch": best_epoch,
        }, checkpoint_path)

    # loss 기록
    def record_loss(self, res, e, train_loss_=None):
        f = open(os.path.join(self.train_params['save_path'], self.train_params['save_name'], 'loss.txt'), 'a')
        if train_loss_ is not None:
            line1 = "[{} epoch] BPRLoss: {:.4f}".format(e, train_loss_)
            f.write(line1 + '\n')
            self.logger.info(line1)
        if self.eval_params['benchmark']:
            line2 = "[{} epoch] B_USER: {}, B_HIT: {}, B_HR: {:.4f}, B_NDCG: {:.4f}".format(
                    e, res['b_global_NUM'], res['b_global_HIT'], res['b_global_HR'], res['b_global_NDCG'])
            f.write(line2 + '\n')
            self.logger.info(line2)
        if self.eval_params['test']:
            line3 = "[{} epoch] T_USER: {}, T_HIT: {}, T_HR: {:.4f}, T_MAP: {:.4f}".format(
                    e, res['t_global_NUM'], res['t_global_HIT'], res['t_global_HR'], res['t_global_MAP'])
            f.write(line3 + '\n')
            self.logger.info(line3)
    
    # mlflow에 loss 기록
    def record_loss_mlflow(self, res, e, cur_lr, train_loss_=None, reg_term_=None):
        if train_loss_ is not None:
            mlflow.log_metric("BPRLoss", train_loss_, e)

        if self.eval_params['benchmark']:
            mlflow.log_metric("B user", res['b_global_NUM'], e)
            mlflow.log_metric("B hit", res['b_global_HIT'], e)
            mlflow.log_metric("B HR", res['b_global_HR'], e)
            mlflow.log_metric("B NDCG", res['b_global_NDCG'], e)
        
        if self.eval_params['test']:
            mlflow.log_metric("T user", res['t_global_NUM'], e)
            mlflow.log_metric("T hit", res['t_global_HIT'], e)
            mlflow.log_metric("T HR", res['t_global_HR'], e)
            mlflow.log_metric("T MAP", res['t_global_MAP'], e)

        mlflow.log_metric("learning rate", cur_lr, e)

        if reg_term_ is not None:
            mlflow.log_metric("Reg Term", reg_term_, e)

    # benchmark process시 result 계산. HR과 NDCG 계산
    def benchmark_process_batch(self, cur_epoch, recent_user_interval=8):
        self.model.eval()
        inter_HR, inter_NDCG = [], []
        
        # benchmark_dl의 batch수 만큼 iterate
        for idx, (pos_neg_pair, history, history_mask) in enumerate(tqdm(self.benchmark_dl)):
            # pos_neg_pair에서 첫 번째 column -> user, 두 번째부터 마지막까지의 column -> pos_neg

            user, pos_neg = pos_neg_pair[:, 0], pos_neg_pair[:, 1:]
            num_negs = pos_neg.shape[1] - 1

            user = user.type(torch.LongTensor).to(self.device)
            pos_neg = pos_neg.type(torch.LongTensor).to(self.device)

            history = history.type(torch.LongTensor).to(self.device)
            history_mask = history_mask.type(torch.LongTensor).to(self.device)

            predictions, inf_user_embed, inf_item_embed = self.model.recommend(user, history, history_mask, item=pos_neg)
            predictions = predictions.detach()
            _, recommends = torch.topk(predictions, self.train_params['top_k'])
            recommends = recommends.detach().cpu().numpy()
            #print("recommends.shape:", recommends.shape)

            recommends_to_idxes = pos_neg.detach().cpu().numpy()[np.arange(recommends.shape[0])[:, None], recommends]
            
            # recent_ten flag가 켜져있으면
            if self.eval_params['recent_ten']:
                users_converted = user.cpu().numpy().astype(np.int32)
                recent_items = []
                for per_user in users_converted:
                    per_user_hist = self.data_dict['user_train_dict'][per_user]

                    if len(per_user_hist) < self.train_params['top_k']:
                        recent_items.append(per_user_hist[::-1] + self.popular_items[:(self.train_params['top_k']- len(per_user_hist))])
                    else:
                        recent_items.append(per_user_hist[::-1][:self.train_params['top_k']])

            users = user.cpu().numpy()
            inter_gt_items, inter_recommends = [], []
            # user수만큼
            for idx, per_user in enumerate(users):
                user_gt_items = [self.data_dict['item2idx'].get(i, -1) for i in self.data_dict['user_last_test_dict'][self.data_dict['idx2user'][per_user]]]
                #print("user_gt_items:", user_gt_items)
                inter_gt_items.append(user_gt_items)
                inter_recommends.append(recommends_to_idxes[idx].tolist())
                #print("inter_gt_itmes:", inter_gt_items)
                #print("inter_recommends:", inter_recommends)
            
            # batch하나의 hit, ndcg 계산해 loop마다 더함
            inter_HR += hit(inter_gt_items, inter_recommends, batch=True)
            inter_NDCG += ndcg(inter_gt_items, inter_recommends, batch=True)
            #print("inter_HR:", inter_HR)
            #print("inter_NDCG:", inter_NDCG)

        res = {}

        global_HR, global_NDCG = inter_HR, inter_NDCG

        res['b_global_NUM'] = len(global_HR)
        res['b_global_HIT'] = int(np.sum(global_HR))
        res['b_global_HR'] = np.mean(global_HR)
        res['b_global_NDCG'] = np.mean(global_NDCG)
        #import pdb; pdb.set_trace()
        return res

    # test process시 result 계산
    def test_process_batch(self, cur_epoch, recent_user_interval=8):
        self.model.eval()
        inter_HR, inter_MAP, test_num_common_recs = [], [], []
        
        # self.test_ds의 batch수 만큼 iterate
        for idx, (user, mask, history, history_mask) in enumerate(tqdm(self.test_dl)):
            user = user[:, 0].type(torch.LongTensor).to(self.device)
            mask = mask.type(torch.FloatTensor).to(self.device)

            history = history.type(torch.LongTensor).to(self.device)
            history_mask = history_mask.type(torch.LongTensor).to(self.device)

            predictions, inf_user_embed, inf_item_embed = self.model.recommend(user, history, history_mask)
            predictions = predictions.detach()
            _, recommends = torch.topk(predictions, self.train_params['top_k'])
            recommends = recommends.detach().cpu().numpy()

            row2row_comp = np.concatenate([np.tile(np.expand_dims(recommends, axis=1),(1, recommends.shape[0], 1)), 
                                           np.tile(np.expand_dims(recommends, axis=0),(recommends.shape[0], 1, 1))], axis=-1)
            common_recs = np.apply_along_axis(lambda x : 20 - len(np.unique(x)), axis=-1, arr = row2row_comp)
            num_common_recs = np.sum(np.triu(common_recs, k=1)) / (np.sum(np.triu(np.ones_like(common_recs), k=1)) + 1e-8)

            test_num_common_recs.append(num_common_recs)

            # recent_ten flag가 켜져있으면 recent_items를 계산해 저장
            if self.eval_params['recent_ten']:
                users_converted = user.cpu().numpy().astype(np.int32)
                recent_items = []
                for per_user in users_converted:
                    per_user_hist = self.data_dict['user_train_dict'][per_user]

                    if len(per_user_hist) < self.train_params['top_k']:
                        recent_items.append(per_user_hist[::-1] + self.popular_items[:(self.train_params['top_k']- len(per_user_hist))])
                    else:
                        recent_items.append(per_user_hist[::-1][:self.train_params['top_k']])

            users = user.cpu().numpy()
            inter_gt_items, inter_recommends = [], []
            
            # user 수만큼 iterate
            for idx, per_user in enumerate(users):
                # user_gt_items = [self.data_dict['item2idx'].get(i, -1) for i in self.data_dict['user_test_dict'][self.data_dict['idx2user'][per_user]]]
                user_gt_items = [self.data_dict['item2idx'].get(i, -1) for i in self.data_dict['user_last_test_dict'][self.data_dict['idx2user'][per_user]]]
                
                inter_gt_items.append(user_gt_items)
                inter_recommends.append(recommends[idx])
            
            # inter_HR, inter_MAP 계산
            inter_HR += hit(inter_gt_items, inter_recommends, batch=True)
            inter_MAP += map_(inter_gt_items, inter_recommends, self.train_params['top_k'], batch=True)

        # test_num_common_recs를 mlflow에 기록
        mlflow.log_metric("test_num_common_recs", np.mean(test_num_common_recs), cur_epoch)

        res = {}

        global_HR, global_MAP = inter_HR, inter_MAP

        res['t_global_NUM'] = len(global_HR)
        res['t_global_HIT'] = int(np.sum(global_HR))
        res['t_global_HR'] = np.mean(global_HR)
        res['t_global_MAP'] = np.mean(global_MAP)

        return res

    def train(self, use_checkpoint=False):
        self.logger.info('Start Training.')
        self.logger.info('Start to log training process.\n')
        self.save_config()

        # os.makedirs(os.path.join(self.train_params['attention_path'], self.train_params['save_name']), exist_ok=True)
        
        # checkpoint 사용시, model parameter와 metric별 best result 불러오고, 사용 하지 않으면 0으로 초기화
        if use_checkpoint:
            self.logger.info('Continue to train from the last checkpoint : {}'.format(use_checkpoint))
            checkpoint_path = os.path.join(self.train_params['save_path'], self.train_params['save_name'], '{:06d}_epoch.pth'.format(use_checkpoint))
            cp = torch.load(checkpoint_path)
            self.model.load_state_dict(cp['state_dict'])
            best_hr, best_ndcg, best_epoch = cp['best_hr'], cp['best_ndcg'], cp['best_epoch']
            start_epoch = use_checkpoint + 1
        else:
            start_epoch = 0
            best_hr, best_ndcg, best_epoch = 0, 0, 0

        # benchmark_res와 test_res의 시작값을 초기화하고 기록
        start_benchmark_res, start_test_res = {}, {}
        if self.eval_params['benchmark']:
            start_benchmark_res = self.benchmark_process_batch(cur_epoch=0)
            print("start_benchmark_res:", start_benchmark_res)
        if self.eval_params['test']:
            start_test_res = self.test_process_batch(cur_epoch=0)
            print("start_test_res:", start_benchmark_res)
        self.record_loss({**start_benchmark_res, **start_test_res}, 'start')
        

        # 첫 번째 for-loop. epoch수 만큼 iterate
        for e in range(start_epoch, start_epoch + self.train_params['train_epoch']):
            self.model.train()
            
            train_batch_loss, att_for_vis = [], []
            reg_batch = []
            
            # 두 번째 for-loop, train_dl의 batch수 만큼 iterate
            for idx, pos_neg_pair in enumerate(tqdm(self.train_dl)):
                neg_history, neg_history_mask = None, None
                
                # train DataLoader로부터 model에 input으로 넣어주는 vector
                user, pos, neg = pos_neg_pair[:, 0], pos_neg_pair[:, 1], pos_neg_pair[:, -1]
                prodcode, prodtype = pos_neg_pair[:, 2], pos_neg_pair[:, 3]
                graph_appear, colour_group, pcolval, pcolmas = pos_neg_pair[:, 4], pos_neg_pair[:, 5], pos_neg_pair[:, 6], pos_neg_pair[:, 7]
                depart, idxgroup, section, garmgroup = pos_neg_pair[:, 7], pos_neg_pair[:, 8], pos_neg_pair[:, 9], pos_neg_pair[:, 10]
                history, history_mask = pos_neg_pair[:, 11:11+self.model_params['seq_len']], pos_neg_pair[:, 11+self.model_params['seq_len']:11+2*self.model_params['seq_len']]
                
                # sampler가 TwoView인 경우 neg_history, neg_history_mask
                if self.model_params['sampler_type'] == "TwoView":
                    neg_history = pos_neg_pair[:, 11+2*self.model_params['seq_len']:11+3*self.model_params['seq_len']]
                    neg_history_mask = pos_neg_pair[:, 11+3*self.model_params['seq_len']:11+4*self.model_params['seq_len']]

                    neg_history = torch.LongTensor(neg_history).to(self.device)
                    neg_history_mask = torch.LongTensor(neg_history_mask).to(self.device)

                if sum(history_mask[:,-1]) < pos_neg_pair.shape[0]:
                    import pdb; pdb.set_trace()
                
                # tensor 값을 int64 value로 변경하고 self.device에 넣음
                user = torch.LongTensor(user).to(self.device)
                pos = torch.LongTensor(pos).to(self.device)
                prodcode = torch.LongTensor(prodcode).to(self.device)
                prodtype = torch.LongTensor(prodtype).to(self.device)
                graph_appear = torch.LongTensor(graph_appear).to(self.device)
                colour_group = torch.LongTensor(colour_group).to(self.device)
                pcolval = torch.LongTensor(pcolval).to(self.device)
                pcolmas = torch.LongTensor(pcolmas).to(self.device)
                depart = torch.LongTensor(depart).to(self.device)
                idxgroup = torch.LongTensor(idxgroup).to(self.device)
                section = torch.LongTensor(section).to(self.device)
                garmgroup = torch.LongTensor(garmgroup).to(self.device)
                neg = torch.LongTensor(neg).to(self.device)
                history = torch.LongTensor(history).to(self.device)
                history_mask = torch.LongTensor(history_mask).to(self.device)
                
                # gradient 초기화
                self.opt.zero_grad() 
                
                # input vector들을 model에 넣고 output 산출
                total_loss, loss, reg, user_history_att, user_out, pos_out, neg_out = self.model(user, pos, prodcode, prodtype, graph_appear, colour_group,
                                                                                                 pcolval, pcolmas, depart, idxgroup, section, garmgroup,
                                                                                                 neg, history, history_mask)
                
                #import pdb; pdb.set_trace()
                
                # gradient descent 계산
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_params['clip_grad_ratio']) # Gradient 최대 개수를 제한하고, gradient가 최대치를 넘으면 gradient 크기 재조정. # Adam opt를 쓰면 필요없으나 안전장치로 넣어줌
                # parameter 조정
                self.opt.step()
                
                # scheduler 조정
                if self.train_params['scheduler_type'] != 'original': 
                    self.scheduler.step()
                
                # train_batch_loss를 array에 저장
                train_batch_loss.append(loss.item())
                # regularization term이 있으면 reg_batch array에 추가
                if reg is not None:
                    reg_batch.append(reg.item())
                else:
                    reg_batch.append(0)
                    
                # attention visual flag가 켜져있으면 att_for_vis array에 user_history_att 추가
                if self.train_params["attention_vis_flag"]:
                    att_for_vis.append(user_history_att[:1, :].detach().cpu().numpy()) # user_history_att[:1, :]가 어떤 값일지?

                # track_additional_info flag가 켜져있고 idx가 interval_additional_info의 배수에 도달할 때마다
                if self.train_params["track_additional_info"] and idx % self.train_params["interval_additional_info"] == 0:
                    raw_user, raw_pos, raw_neg = user_out.detach().cpu().numpy(), pos_out.detach().cpu().numpy(), neg_out.detach().cpu().numpy() # .detach().cpu().numpy(): tensor를 gpu에서 분리해서 cpu로 올린 뒤, numpy로 변환해 반환

                    # numpy vector normalization. 1e-8을 붙여주는 이유는?
                    user_out = raw_user / (np.linalg.norm(raw_user, axis=-1, keepdims=True) + 1e-8)
                    pos_out = raw_pos / (np.linalg.norm(raw_pos, axis=-1, keepdims=True) + 1e-8)
                    neg_out = raw_neg / (np.linalg.norm(raw_neg, axis=-1, keepdims=True) + 1e-8)
                    
                    # matrix간 similarity 계산. 조금 후에 다시보자
                    user_sim, user_sim_above9 = self._calculate_similarity(user_out)
                    pos_sim, pos_sim_above9 = self._calculate_similarity(pos_out)
                    neg_sim, neg_sim_above9 = self._calculate_similarity(neg_out)

                    # mlflow에 log 기록
                    mlflow.log_metric("train_u2u_cos", user_sim, idx // self.train_params["interval_additional_info"])
                    mlflow.log_metric("train_u2p_cos", np.multiply(user_out, pos_out).sum(axis=-1).mean(), idx // self.train_params["interval_additional_info"])
                    mlflow.log_metric("train_u2n_cos", np.multiply(user_out, neg_out).sum(axis=-1).mean(), idx // self.train_params["interval_additional_info"])
                    mlflow.log_metric("train_p2n_cos", np.multiply(pos_out, neg_out).sum(axis=-1).mean(), idx // self.train_params["interval_additional_info"])
                    mlflow.log_metric("train_p2p_cos", pos_sim, idx // self.train_params["interval_additional_info"])
                    mlflow.log_metric("train_n2n_cos", neg_sim, idx // self.train_params["interval_additional_info"])

                    mlflow.log_metric("v_user_sim_above9", user_sim_above9, idx // self.train_params["interval_additional_info"])
                    mlflow.log_metric("v_pos_sim_above9", pos_sim_above9, idx // self.train_params["interval_additional_info"])
                    mlflow.log_metric("v_neg_sim_above9", neg_sim_above9, idx // self.train_params["interval_additional_info"])

                    mlflow.log_metric("z_user_norm", np.mean(np.linalg.norm(raw_user, axis=-1)), idx // self.train_params["interval_additional_info"])
                    mlflow.log_metric("z_pos_norm", np.mean(np.linalg.norm(raw_pos, axis=-1)), idx // self.train_params["interval_additional_info"])
                    mlflow.log_metric("z_neg_norm", np.mean(np.linalg.norm(raw_neg, axis=-1)), idx // self.train_params["interval_additional_info"])

            # train_batch_loss, reg_batch array로부터 평균을 구해 train_loss, reg_term에 저장
            train_loss_ = np.array(train_batch_loss).mean()
            reg_term_ = np.array(reg_batch).mean()

            benchmark_res, test_res = {}, {}
            if self.eval_params['benchmark']:
                benchmark_res = self.benchmark_process_batch(cur_epoch=e+1)
            if self.eval_params['test']:
                test_res = self.test_process_batch(cur_epoch=e+1)
            total_res = {**benchmark_res, **test_res}
            
            if self.train_params['scheduler_type'] != 'original':
                cur_lr = self.scheduler.get_last_lr()
            else:
                cur_lr = [self.opt.param_groups[0]['lr']]

            self.record_loss_mlflow(total_res, e, cur_lr[0], train_loss_, reg_term_)
            self.record_loss(total_res, e, train_loss_)

            if self.train_params["attention_vis_flag"]:
                img_for_vis = np.vstack(att_for_vis)

                fig = plt.figure(figsize=(10, 10))
                for i in range(1, 10 + 1):
                    fig.add_subplot(10, 1, i)
                    plt.imshow(img_for_vis[i:i+1])

                plt.savefig(os.path.join(self.train_params['attention_path'], self.train_params['save_name'],'attention_map_{}.png'.format(e)))
                plt.close(fig)
            
            if total_res['b_global_HR'] > best_hr:
                best_hr, best_ndcg, best_epoch = total_res['b_global_HR'], total_res['b_global_NDCG'], e
            self.save_checkpoint(e, best_hr, best_ndcg, best_epoch, self.model)

        self.logger.info("Training Completed. Best epoch {:03d}: GLOBAL_HR = {:.4f}, GLOBAL_NDCG = {:.4f}".format(best_epoch, best_hr, best_ndcg))

        mlflow.end_run()

    def _calculate_similarity(self, matrix):
        normalized_matrix = matrix / (np.linalg.norm(matrix, axis=-1, keepdims=True) + 1e-8)

        # sim_matrix = (np.expand_dims(normalized_matrix, 1) * normalized_matrix).sum(axis=-1)
        sim_matrix = (np.tile(np.expand_dims(normalized_matrix, axis=1), (1, normalized_matrix.shape[0], 1)) * 
                      np.tile(np.expand_dims(normalized_matrix, axis=0), (normalized_matrix.shape[0], 1, 1))).sum(axis=-1)

        triu_matrix = np.triu(sim_matrix, k=1)

        return (np.sum(triu_matrix) / (np.sum(np.triu(np.ones_like(triu_matrix), k=1).astype(np.int32)) + 1e-8), 
                np.sum(np.where(triu_matrix > 0.9, 1, 0).astype(np.int32)) / (np.sum(np.triu(np.ones_like(triu_matrix), k=1).astype(np.int32)) + 1e-8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/config.json', help='Model train configuration file path')
    parser.add_argument('--use_checkpoint', type=int, default=False, help='Continue to train with specified checkpoint number.')
    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    # Initialize trainer
    t = Trainer(config_path=args.config_path)
    
    mlflow.set_tracking_uri(t.config['mlflow_params']['remote_server_uri'])
    t.logger.info("Connect to mlflow : {}".format(mlflow.tracking.get_tracking_uri()))
    mlflow.set_experiment(t.config['mlflow_params']['experiment_name'])
    run = mlflow.start_run(run_name=t.config['train_params']['save_name'])
    mlflow.log_params(t.config['model_params'])
    mlflow.log_params(t.config['train_params'])

    t.train(use_checkpoint=args.use_checkpoint)