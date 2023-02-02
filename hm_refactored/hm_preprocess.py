import os
import sys

import json
import pickle
import argparse
import warnings
from tqdm import tqdm

import pandas as pd
import numpy as np
import copy
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from util.helper import init_logger

tqdm.pandas()
warnings.filterwarnings('ignore')
logger = init_logger('preprocessor')

def hm_prep(config):
    
    # 0. path 설정
    train_data_name = config['train_data_name']
    test_data_name = config['test_data_name']
    orig_path = config['orig_path']
    save_path = os.path.join(config['dataset_path'], config['save_name'] + '.pkl')
    target_day = config['target_day']
    reset = config['reset']

    logger.info('Read data files.')
    train_data_type, test_data_type = train_data_name[-3:], test_data_name[-3:]
    
    train_data_path = os.path.join(orig_path, train_data_name)
    test_data_path = os.path.join(orig_path, test_data_name)

    load_funcs = {'csv': (lambda x: pd.read_csv(x)), 'pkl': (lambda x: pickle.load(open(x, 'rb')))}
    # import pdb; pdb.set_trace()
    # 1. column 선택 및 정렬, train_target_df, test_target_df 설정
    train_data = load_funcs[train_data_type](train_data_path)
    
    train_data = train_data[['user_id', 'item_id', 'timestamp', 'count', 'occurence']]
    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
    
    train_num_days = (train_data['timestamp'].max() - train_data['timestamp'].min()).days
    train_data['day'] = train_num_days - (train_data['timestamp'].max() - train_data['timestamp']).dt.days
    #train_data = train_data.drop_duplicates(subset=['user_id', 'item_id', 'day'], keep='last') # 31788324 -> 28575395 rows
    #train_data = train_data.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])
    #train_data = train_data.reset_index(drop=True)

    logger.info('Split train / test.')
    target_columns = ['user_id', 'item_id', 'timestamp', 'day', 'count', 'occurence']
    train_target_df = train_data[target_columns]
    test_target_df = train_data[target_columns] # 동일한 transactions_train.csv를 쓰므로 train_target_df와 동일하게 설정
    
    # 2. target_day 이전 날짜와 target_day 해당 날짜의 데이터를 분리해 각각 train_df, test_df에 저장
    if config['recent_two_weeks']:
        train_df = train_target_df[(train_target_df['day'] < target_day) & (train_target_df['day'] > (target_day - 15))] # 28575395 -> 534352 rows
    else:
        train_df = train_target_df[train_target_df['day'] < target_day]
    test_df = test_target_df[test_target_df['day'] == target_day] # 53280 rows

    # 3. train_df에 count와 occurence column추가
    if config['slice_user_by_count']:
        #tmp = train_df.groupby('user_id')['item_id'].count()
        #count = tmp.reset_index().rename(columns={"item_id": "count"})
        #occurence = tmp.apply(lambda x : np.arange(x)).explode('item_id').rename("occurence")
        
        #train_df = train_df.merge(count, on='user_id', how='left')
        #train_df = pd.concat([train_df, occurence], axis=1)
        
        # train_df = train_df[train_df['count'] < config['count_high']][train_df['count'] > config['count_low']]
        train_df = train_df[(train_df['count'] < config['count_high']) & (train_df['count'] > config['count_low'])] # 499120 rows
    
    # 4. train_df/test_df의 last_occur을 정의하고 last_test_df, unique_last_test_df 정의
    train_last_occur = (train_df.groupby(by='user_id')['timestamp'].transform(max) == train_df['timestamp'])
    last_train_df = train_df[train_last_occur] 
    test_last_occur = (test_df.groupby(by='user_id')['timestamp'].transform(max) == test_df['timestamp'])
    last_test_df = test_df[test_last_occur] 
    unique_last_test_df = last_test_df.drop_duplicates(['user_id']) # 53280 rows -> 16092 rows

    # 5. user_id/item_id dtype을 category로 변경
    train_df['user_id'] = train_df['user_id'].astype('category')
    train_df['item_id'] = train_df['item_id'].astype('category')
    test_df['user_id'] = test_df['user_id'].astype('category')
    test_df['item_id'] = test_df['item_id'].astype('category') 
    unique_last_test_df['user_id'] = unique_last_test_df['user_id'].astype('category')
    unique_last_test_df['item_id'] = unique_last_test_df['item_id'].astype('category')
    
    combined_user_ids = copy.deepcopy(train_df['user_id']).append(unique_last_test_df['user_id'], ignore_index=True).astype('category')
    combined_item_ids = copy.deepcopy(train_df['item_id']).append(unique_last_test_df['item_id'], ignore_index=True).astype('category')

    # 6. user/item dictionary 생성, interaction과 mapping
    logger.info('Make user item index dictionary & map to interactions.')
    
    # 6-1. user, item idx mapping set 생성
    train_user2idx = {user_id: idx for idx, user_id in enumerate(train_df['user_id'].cat.categories)} 
    train_item2idx = {item_id: idx for idx, item_id in enumerate(train_df['item_id'].cat.categories)}
    user2idx = {user_id: idx for idx, user_id in enumerate(combined_user_ids.cat.categories)}
    idx2user = {idx: user_id for user_id, idx in user2idx.items()}
    item2idx = {item_id: idx for idx, item_id in enumerate(combined_item_ids.cat.categories)}
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}
    
    # 6-2. user_id, item_id를 각각 user_idx, item_idx로 매핑
    train_df['user_id'] = train_df['user_id'].apply(lambda x: user2idx[x])
    train_df['item_id'] = train_df['item_id'].apply(lambda x: item2idx[x])
    
    # 6-3. cold_user와 cold_item 제거
    if config['remove_cold_user']:
        test_df['cold_user_idx'] = test_df['user_id'].apply(lambda x: train_user2idx.get(x, -1))
        test_df = test_df[test_df['cold_user_idx'] != -1] 
        
        unique_last_test_df['cold_user_idx'] = unique_last_test_df['user_id'].apply(lambda x: train_user2idx.get(x, -1))
        unique_last_test_df = unique_last_test_df[unique_last_test_df['cold_user_idx'] != -1]

    if config['remove_cold_item']:
        test_df['cold_item_idx'] = test_df['item_id'].apply(lambda x: train_item2idx.get(x, -1))
        test_df = test_df[test_df['cold_item_idx'] != -1]

        unique_last_test_df['cold_item_idx'] = unique_last_test_df['item_id'].apply(lambda x: train_item2idx.get(x, -1))
        unique_last_test_df = unique_last_test_df[unique_last_test_df['cold_item_idx'] != -1]

    num_item = len(item2idx)
    num_user = len(user2idx)
    
    # 6-4. train/test에 사용할 Dictionary 생성
    user_train_dict = train_df.sort_values(by='timestamp').groupby('user_id')['item_id'].apply(list).to_dict()
    user_test_dict = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    user_last_test_dict = unique_last_test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    item_train_dict = train_df.groupby('item_id')['user_id'].apply(list).to_dict()
    
    #import pdb; pdb.set_trace()

    # 7. zero history, recent_bought항목 제거
    if config['remove_zero_history']:
        idx = train_df.groupby(['user_id'])['timestamp'].transform(min) != train_df['timestamp']
        train_df = train_df[idx]
    
    if config['remove_recent_bought']:
        unique_last_test_df['recent_bought'] = unique_last_test_df.apply(lambda row: item2idx.get(row['item_id']) in user_train_dict[user2idx.get(row['user_id'])], axis=1) 
        unique_last_test_df = unique_last_test_df[unique_last_test_df['recent_bought'] == False]

    if config['remove_train_recent_bought']:
        train_df['recent_bought'] = train_df.apply(lambda row: row['item_id'] in user_train_dict[row['user_id']][:int(row['occurence'])], axis=1)
        train_df = train_df[train_df['recent_bought'] == False]

    inter_last_ts = train_df['timestamp'].max()
    train_df['interval'] = ((inter_last_ts - train_df['timestamp']) / np.timedelta64(1, 'D')).astype(int) + 1
    user_min_interval = train_df[['user_id', 'interval']].groupby('user_id')['interval'].min().to_dict()
    
    # 8. data_dict에 parameter추가
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_name',
        type=str,
        default='transactions_train.csv',
        help='File name of train data.'
    )
    parser.add_argument(
        '--test_data_name',
        type=str,
        default='transactions_train.csv',
        help='File name of test data.'
    )
    parser.add_argument(
        '--orig_path',
        type=str,
        default="/home/omnious/workspace/hyeongseok/datasets/hm",
        help='Folder path that original file saved.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default="/home/omnious/workspace/hyeongseok/datasets/hm/prep",
        help='Folder name of preprocessed file.'
    )
    parser.add_argument(
        '--save_name',
        type=str,
        default='Debug',
        help='File name of preprocessed file.'
    )
    parser.add_argument(
        '--target_day',
        type=int,
        default=32,
        help='Target week of preprocessing.'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='If true, save outputs even though saved files already exist.'
    )
    parser.add_argument(
        '--remove_cold_user',
        action='store_true',
        help='If true, remove cold user from test.'
    )
    parser.add_argument(
        '--remove_cold_item',
        action='store_true',
        help='If true, remove cold item from test.'
    )
    parser.add_argument(
        '--slice_user_by_count',
        action='store_true',
        help='If true, remove cold item from test.'
    )
    parser.add_argument(
        '--count_high',
        type=int,
        default=200,
        help='High limit of slicing index.'
    )
    parser.add_argument(
        '--count_low',
        type=int,
        default=1,
        help='Low limit of slicing index.'
    )
    parser.add_argument(
        '--remove_zero_history',
        action='store_true',
        help='If true, remove user item interaction with zero history.'
    )
    parser.add_argument(
        '--recent_two_weeks'
    )
    args = parser.parse_args()

    config = {}
    config['train_data_name'] = args.train_data_name
    config['test_data_name'] = args.test_data_name
    config['orig_path'] = args.orig_path
    config['dataset_path'] = args.dataset_path
    config['save_name'] = args.save_name
    config['target_day'] = args.target_day
    config['reset'] = args.reset
    config['remove_cold_user'] = args.remove_cold_user
    config['remove_cold_item'] = args.remove_cold_item
    config['slice_user_by_count'] = args.slice_user_by_count
    config['count_high'] = args.count_high
    config['count_low'] = args.count_low
    config['remove_zero_history'] = args.remove_zero_history
    config['recent_two_weeks'] = args.recent_two_weeks
    config = {}
    hm_prep(
        config
    ) 