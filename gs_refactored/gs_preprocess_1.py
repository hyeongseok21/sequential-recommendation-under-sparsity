import os
import sys

import json
import pickle # list, dict와 같은 python object를 그 형태 그대로 저장하고, 불러올 수 있게끔 하는 패키지
import argparse
import warnings
from tqdm import tqdm

import pandas as pd
import numpy as np
import copy #
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))) #getcwd: get current working directory
from util.helper import init_logger

tqdm.pandas()
warnings.filterwarnings('ignore')
logger = init_logger('preprocessor')

def gs_prep(config):
    train_data_name = config['train_data_name']
    test_data_name = config['test_data_name']
    orig_path = config['orig_path']
    save_path = os.path.join(config['dataset_path'], config['save_name'] + '.pkl')
    target_day = config['target_day']
    reset = config['reset']
    
    logger.info('Read data files.')
    train_data_type, test_data_type = train_data_name[-3:], test_data_name[-3:] # (e.g., .csv)
    
    train_data_path = os.path.join(orig_path, train_data_name)
    test_data_path = os.path.join(orig_path, test_data_name)
    
    load_funcs = {'csv': (lambda x: pd.read_csv(x)), 'pkl': (lambda x: pickle.load(open(x, 'rb')))} # 
    
    train_data = load_funcs[train_data_type](train_data_path)
    train_data = train_data.fillna({'dealno': 0.0}) # 결측값 처리: NA -> 0.0
    
    if 'cart' in train_data_name:
        train_data = train_data[['pcid', 'prdid', 'dealno', 'siteid', 'sessionid', 'cartadddtm']]
        train_data.columns = ['user_id', 'item_id', 'deal_no', 'site_id', 'session_id', 'timestamp']
        train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], unit='ms')
    elif 'order' in train_data_name:
        train_data = train_data[['pcid', 'prdid', 'dealno', 'siteid', 'session', 'visitedtime']]
        train_data.columns = ['user_id', 'item_id', 'deal_no', 'site_id', 'session_id', 'timestamp']
        train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], unit='ms')
    train_num_days = (train_data['timestamp'].max() - train_data['timestamp'].min()).days
    train_data['day'] = train_num_days - (train_data['timestamp'].max() - train_data['timestamp']).dt.days
    train_data = train_data.drop_duplicates(subset=['user_id', 'item_id', 'day'], keep='last')
    train_data = train_data.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])
    train_data = train_data.reset_index(drop=True)
    
    if train_data_name == test_data_name:
        test_data = train_data
    else:
        test_data = load_funcs[test_data_type](test_data_path)
        test_data = test_data.fillna({'dealno' : 0.0})
        if 'cart' in test_data_name:
            test_data = test_data[['pcid', 'prdid', 'dealno', 'siteid', 'sessionid', 'cartadddtm']]
            test_data.columns = ['user_id', 'item_id', 'deal_no', 'site_id', 'session_id', 'timestamp']
            test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
        elif 'order' in test_data_name:
            test_data = test_data[['pcid', 'prdid', 'dealno', 'siteid', 'session', 'visitedtime']]
            test_data.columns = ['user_id', 'item_id', 'deal_no', 'site_id', 'session_id', 'timestamp']
            test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], unit='ms')
        test_num_days = (test_data['timestamp'].max() - test_data['timestamp'].min()).days
        test_data['day'] = test_num_days - (test_data['timestamp'].max() - test_data['timestamp']).dt.days
        test_data = test_data.drop_duplicates(subset=['user_id', 'item_id', 'day'], keep='last')
        test_data = test_data.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])
        test_data = test_data.reset_index(drop=True)
        
        logger.info('Split train / test.')
        target_columns = ['user_id', 'item_id', 'timestamp', 'day']
        train_target_df = train_data[target_columns]
        test_target_df = test_data[target_columns]
        
        if config["recent_two_weeks"]:
            train_df = train_target_df[(train_target_df['day'] < target_day) & (train_target_df['day'] > (target_day - 15))]
        else:
            train_df = train_target_df[train_target_df['day'] < target_day]
        test_df = test_target_df[test_target_df['day'] == target_day]
        
        if config['slice_user_by_count']:
            tmp = train_df.groupby('user_id')['item_id'].count()
            count = tmp.reset_index().rename(columns={"item_id": "count"})
            occurence = tmp.apply(lambda x : np.arrange(x).explode('item_id').rename("occurance"))
            
            train_df = train_df.merge(count, on='user_id', how='left')
            train_df = pd.concat([train_df, occurence], axis=1)
            
            train_df = train_df[(train_df['count'] < config['count_high']) & (train_df['count'] > config['count_low'])]
        
        train_last_occur = ()
        last_train_df = train_df[train_last_occur]
        test_last_occur = ()
        last_test_df = test_df[test_last_occur]
        unique_last_test_df = last_test_df.drop_duplicates(['user_id'])
        
        
        
        
        
        
        logger.info('Make user item index dictionary & map to interactions.')
        train_user2idx = {user_id: idx for idx, user_id in enumerate(train_df['user_id'].cat.categories)}
        train_item2idx = {item_id: idx for idx, item_id in enumerate(train_df['item_id'].cat.categories)}
        
        
        
        
        
        
        
        
        if config['remove_cold_user']:
            test_df['cold_user_idx'] = test_df['user_id'].apply(lambda x: train_user2idx.get(x, -1))
            test_df = test_df[test_df['cold_user_idx'] != -1]
        
        if config['remove_cold_item']:
            test_df['cold_item_idx'] = test_df['item_id'].apply(lambda x: train_item2idx.get(x, -1))
            test_df = test_df[test_df['cold_item_idx'] != -1]
                   
        num_item = len(item2idx)
        num_user = len(user2idx)
        
        user_train_dict = train_df.sort_values(by='timestamp')
        
        if config['remove_zero_history']:
            idx = train_df.groupby()
            train_df = train_df[idx]
        
        if config['remove_recent_bought']:
            unique_last_test_df['recent_bought']
        
        if config['remove_train_recent_bought']:
            train_df['recent_bought'] = train_df.apply()
        
        inter_last_ts = train_df['timestamp'].max()
        
        data_dict = {}
        data_dict['train_df'] = train_df
        data_dict['test_df'] = test_df
        data_dict['unique_last_test_df'] = unique_last_test_df
        data_dict['num_user'] = num_user
        data_dict['num_item'] = num_item
        data_dict['user2idx'] = user2idx
        data_dict['idx2user'] = idx2user
        data_dict['item2idx'] = item2idx
        data_dict['idx2item'] = idx2item
        data_dict['user_train_dict'] = user_train_dict
        data_dict['user_test_dict'] = user_test_dict
        data_dict['user_last_test_dict'] = user_last_test_dict
        data_dict['item_train_dict'] = item_train_dict
        data_dict['user_min_interval'] = user_min_interval
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        return data_dict