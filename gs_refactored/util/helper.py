import os
import six
import math
import copy
import pickle
import logging
from glob import glob
from datetime import datetime

import cv2
from PIL import Image
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .metric import hit, map_


def init_logger(logger_name, level='INFO'):
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', datefmt='%I:%M:%S:%p', level=logging.INFO)
    logger = logging.getLogger(logger_name)
    if level != 'INFO':
        logger.setLevel(logging.ERROR)
    return logger

def parse_time(x):
    d = datetime.fromtimestamp(x)
    return str(d.year) + '{:02d}'.format(d.month) + '{:02d}'.format(d.day)

def parse_time_2(x):
    return ''.join(x.split(' ')[0].split('-'))

def get_loss_df(bp, cp_lst):
    df_lst = []
    for cp in cp_lst:
        p = os.path.join(bp, cp)
        kws = p.split('/')[-1]
        with open(os.path.join(p, 'loss.pkl'), 'rb') as f:
            arr = pickle.load(f)
            arr = np.array(arr).T
        df = pd.DataFrame(arr, columns=['HR', 'NDCG', 'TCE'])
        df = df.reset_index()
        
        df.columns = ['epoch', 'HR', 'NDCG', 'TCE']
        df['exp'] = kws
        
        df_lst.append(df)
    
    df = pd.concat(df_lst, axis=0).reset_index(drop=True)
    
    max_df = df.copy()
    max_df['HR'] = max_df.groupby(['model', 'exp'])['HR'].transform(np.max)
    max_df['NDCG'] = max_df.groupby(['model', 'exp'])['NDCG'].transform(np.max)
    max_df = max_df[['HR', 'NDCG', 'model', 'exp']].drop_duplicates().reset_index(drop=True)
    
    return df, max_df

def draw_items(item_id_lst, item_label_lst, idx2item=None, size=256):
    images_path = '/data/shared2/reco_datasets/hnm/images'
    images_path_dict = {int(i.split('/')[-1].split('.')[0]): i for i in glob(os.path.join(images_path, '*.jpg'))}
    
    if idx2item is not None:
        images_idx = [idx2item[idx] for idx in item_id_lst]
    else:
        images_idx = item_id_lst
    images_path = [images_path_dict.get(idx, -1) for idx in images_idx]
    if len(images_path) % 5 != 0:
        images_path += ([-1] * (5 - len(images_path) % 5))
        item_label_lst += ([-1] * (5 - len(images_path) % 5))
    images_lst = []
    for idx, p in enumerate(images_path):
        if p == -1:
            images_lst.append(np.ones((size, size, 3)) * 255)
        else:
            img = cv2.imread(p)
            img = cv2.resize(img, (size, size))
            img = cv2.putText(
                img=img,
                text=str(item_label_lst[idx]),
                org=(5, 240),
                fontFace=0,
                fontScale=0.7,
                color=(119, 203, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            images_lst.append(img)
    image_grid = np.concatenate(images_lst, axis=1)[:, :, ::-1]
    grid_lst = []
    prev = 0
    for i in range(1, (len(images_lst) // 5) + 1):
        tmp_size = size * (5 * i)
        tmp_grid = image_grid[:, prev:tmp_size]
        grid_lst.append(tmp_grid)
        prev = tmp_size
    image_grid = np.concatenate(grid_lst, axis=0)
    image_grid = image_grid.astype(np.uint8)
    Image.fromarray(image_grid).show()

def vis_user_rec(self, user_result, user_idx, size=256):
    """
    Visualize user's 10 interactions(randomly sampled) and recommendations with real images.
    It also contains item's meta informations.
    """
    assert self.data_dict['item_meta'] is not None
    item_df = self.data_dict['item_meta']
    user_train_data = self.data_dict['user_train_dict'].get(user_idx, [])
    try:
        user_train_10 = np.random.choice(user_train_data, 10, replace=False)
    except:
        user_train_10 = user_train_data + ([-1] * (10 - len(user_train_data)))


    self.logger.info("User interacted items. # Interactions : {}".format(len(user_train_data)))
    # item_user_count = [len(self.data_dict['item_user_count_dict'][self.idx2item[idx]]) if idx != -1 else idx for idx in user_train_10]
    images_idx = [self.data_dict['idx2item'][idx] if idx != -1 else idx for idx in user_train_10]
    images_path = [self.images_path[idx] if idx != -1 else idx for idx in images_idx]
    images_lst = []
    for idx, p in enumerate(images_path):
        if p == -1:
            images_lst.append(np.ones((size, size, 3)) * 255)
        else:
            item_info = item_df[item_df['item_id'] == images_idx[idx]]
            item_name, item_cate, ts = item_info[['item_name', 'category', 'timestamp']].values[0]
            self.logger.info('[{} / {}] RTS:{}\tITEM:{}'.format(idx+1, item_cate, ts, item_name))

            img = cv2.imread(p)
            img = cv2.resize(img, (size, size))
            img = cv2.putText(
                img=img,
                text=str(idx),
                org=(5, 240),
                fontFace=0,
                fontScale=0.7,
                color=(119, 203, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            images_lst.append(img)
    image_grid = np.concatenate(images_lst, axis=1)[:, :, ::-1]
    image_grid_1 = image_grid[:, :size*5, :]
    image_grid_2 = image_grid[:, size*5:, :]
    image_grid = np.concatenate([image_grid_1, image_grid_2], axis=0)
    image_grid = image_grid.astype(np.uint8)
    Image.fromarray(image_grid).show()


    self.logger.info("User topk recommendations".format(int(user_idx)))
    user_result = user_result[user_idx]
    user_gt = user_result['gt']
    top_k_result = user_result['rec']
    top_k_result = [user_gt] + top_k_result
    # item_user_count = [len(self.data_dict['item_user_count_dict'][self.idx2item[idx]]) if idx != -1 else idx for idx in top_k_result]
    images_idx = [self.data_dict['idx2item'][idx] for idx in top_k_result]
    images_path = [self.images_path[idx] for idx in images_idx]
    is_hit = user_result['hit']
    images_lst = []
    for idx, p in enumerate(images_path):
        item_info = item_df[item_df['item_id'] == images_idx[idx]]
        item_name, item_cate, ts = item_info[['item_name', 'category', 'timestamp']].values[0]
        if idx == 0:
            t = 'gt'
            self.logger.info('[{} / {}] RTS:{}\tITEM:{}'.format(t, item_cate, ts, item_name))
        else:
            t = idx
            self.logger.info('[{} / {}] RTS:{}\tITEM:{}'.format(idx, item_cate, ts, item_name))

        img = cv2.imread(p)
        img = cv2.resize(img, (size, size))
        img = cv2.putText(
            img=img,
            text=str(t),
            org=(5, 240),
            fontFace=0,
            fontScale=0.7,
            color=(119, 203, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        images_lst.append(img)
        if idx == 0:
            for i in range(4):
                images_lst.append(np.ones((size, size, 3)) * 255)

    image_grid = np.concatenate(images_lst, axis=1)[:, :, ::-1]
    image_grid_1 = image_grid[:, :size*5, :]
    image_grid_2 = image_grid[:, size*5:size*10, :]
    image_grid_3 = image_grid[:, size*10:, :]
    image_grid = np.concatenate([image_grid_1, image_grid_2, image_grid_3], axis=0)
    image_grid = image_grid.astype(np.uint8)
    Image.fromarray(image_grid).show()

def vis_embeddings(embedding_matrix):
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(embedding_matrix)
    pca_df = pd.DataFrame(coords_pca, columns=['x', 'y'])

    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 10)
    
    ax.set_title('[PCA] embedding visualization', fontsize=20)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    graph = sns.scatterplot(data=pca_df, x='x', y='y', ax=ax)
    plt.show()


def generate_config(config, params_dict, param_combs=None):
    def select(params, i=0, tmp=[]):
        if i > (len(params) - 1):
            return tmp
        else:
            for p in params[i]:
                res = select(params, i=i+1, tmp=tmp+[p])
                if res is not None:
                    param_combs.append(res)
    if param_combs is None:
        param_combs = []
        select(list(params_dict.values()))

    configs = []
    for comb in param_combs:
        tmp_config = copy.deepcopy(config)
        for kv, c in zip(params_dict.keys(), comb):
            key, value = kv.split(':')
            tmp_config[key][value] = c
        configs.append(tmp_config)
    return configs


def evaluate(user_rec_dict, items_for_cold_users, recent_user_interval=15, top_k=12, get_result=False):
    # Process of pickled data.

    # data_path = '/data/shared2/reco_datasets/hm'
    # user_meta = pd.read_csv(os.path.join(data_path, 'customers.csv'))
    # item_meta = pd.read_csv(os.path.join(data_path, 'articles.csv'))
    # interactions = pd.read_csv(os.path.join(data_path, 'transactions_train.csv'))
    # submit = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
    # user_meta = user_meta.fillna('NONE')
    # user_meta.columns = ['user_id'] + list(user_meta.columns[1:])
    # columns = ['article_id', 'product_code'] + [col for col in item_meta.columns if 'name' in col]
    # item_meta = item_meta[columns]
    # item_meta.columns = ['item_id'] + list(item_meta.columns[1:])
    # interactions = interactions[['customer_id', 'article_id', 't_dat', 'price', 'sales_channel_id']]
    # interactions.columns = ['user_id', 'item_id', 'timestamp', 'price', 'sales_channel_id']
    # interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    # common_test = interactions[interactions['timestamp'] >= '2020-09-16']
    # common_train = interactions.loc[~interactions.index.isin(common_test.index)]
    # common_train['interval'] = ((common_train['timestamp'].max() - common_train['timestamp']) / np.timedelta64(1, 'D')).astype(int)
    # c_user_min_interval = common_train[['user_id', 'interval']].groupby('user_id')['interval'].min().to_dict()
    # c_user_test_dict = common_test.groupby('user_id')['item_id'].apply(list).to_dict()
    
    with open('/data/shared2/reco_datasets/hnm/test/common_test_interval_info.pkl', 'rb') as f:
        user_test_dict, user_min_interval = pickle.load(f)
    
    recent_HR, recent_MAP, inter_HR, inter_MAP, new_HR, new_MAP = [], [], [], [], [], []
    
    recent_gt_items, recent_recommends = [], []
    inter_gt_items, inter_recommends = [], []
    new_gt_items, new_recommends = [], []
    
    for u in list(user_test_dict.keys()):
        if user_rec_dict.get(u, -1) == -1:
            user_rec_dict[u] = items_for_cold_users[:top_k]
        min_interval = user_min_interval.get(u, -1)
        if min_interval != -1:
            if min_interval <= recent_user_interval:
                recent_gt_items.append(user_test_dict[u])
                recent_recommends.append(user_rec_dict[u])
            else:
                inter_gt_items.append(user_test_dict[u])
                inter_recommends.append(user_rec_dict[u])
        else:
            new_gt_items.append(user_test_dict[u])
            new_recommends.append(user_rec_dict[u])

    recent_HR += hit(recent_gt_items, recent_recommends, batch=True)
    recent_MAP += map_(recent_gt_items, recent_recommends, top_k, batch=True)
    inter_HR += hit(inter_gt_items, inter_recommends, batch=True)
    inter_MAP += map_(inter_gt_items, inter_recommends, top_k, batch=True)
    new_HR += hit(new_gt_items, new_recommends, batch=True)
    new_MAP += map_(new_gt_items, new_recommends, top_k, batch=True)
    
    res = {}
    for k, v in zip(['recent', 'inter', 'new'], [[recent_HR, recent_MAP], [inter_HR, inter_MAP], [new_HR, new_MAP]]):
        num_user = len(v[0])
        res['{}_{}'.format(k, 'NUM')] = num_user
        if num_user != 0:
            res['{}_{}'.format(k, 'HIT')] = int(np.sum(v[0]))
            res['{}_{}'.format(k, 'HR')] = np.mean(v[0])
            res['{}_{}'.format(k, 'MAP')] = np.mean(v[1])
        else:
            res['{}_{}'.format(k, 'HIT')] = 0
            res['{}_{}'.format(k, 'HR')] = 0
            res['{}_{}'.format(k, 'MAP')] = 0
    global_HR = recent_HR + inter_HR + new_HR
    global_MAP = recent_MAP + inter_MAP + new_MAP
    res['global_NUM'] = len(global_HR)
    res['global_HIT'] = int(np.sum(global_HR))
    res['global_HR'] = np.mean(global_HR)
    res['global_MAP'] = np.mean(global_MAP)
    
    if get_result:
        return res
    else:
        print("[Rec result] RECENT USER: {}, RECENT_HIT: {}, RECENT_HR: {:.4f}, RECENT_MAP: {:.4f}".format(
            res['recent_NUM'], res['recent_HIT'], res['recent_HR'], res['recent_MAP'])
        )
        print("[Rec result] INTER USER: {}, INTER_HIT: {}, INTER_HR: {:.4f}, INTER_MAP: {:.4f}".format(
            res['inter_NUM'], res['inter_HIT'], res['inter_HR'], res['inter_MAP'])
        )
        print("[Rec result] NEW USER: {}, NEW_HIT: {}, NEW_HR: {:.4f}, NEW_MAP: {:.4f}".format(
            res['new_NUM'], res['new_HIT'], res['new_HR'], res['new_MAP'])
        )
        print("[Rec result] GLOBAL USER: {}, GLOBAL_HIT: {}, GLOBAL_HR: {:.4f}, GLOBAL_MAP: {:.4f}".format(
            res['global_NUM'], res['global_HIT'], res['global_HR'], res['global_MAP'])
        )
        return