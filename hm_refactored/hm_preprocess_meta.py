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

def hm_prep_meta(config):
    
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

    # 1. column 선택 및 정렬, train_target_df, test_target_df 설정
    train_data = load_funcs[train_data_type](train_data_path)
    
    train_data = train_data[['user_id', 'age', 'postal_code', 'item_id', 'price', 'timestamp', 'count', 'occurence', 'product_code', 'product_type_no', 'graphical_appearance_no', 
                             'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_group_no', 'section_no', 'garment_group_no']]
    # train_data.columns = ['user_id', 'item_id', 'timestamp']
    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
    
    train_num_days = (train_data['timestamp'].max() - train_data['timestamp'].min()).days
    train_data['day'] = train_num_days - (train_data['timestamp'].max() - train_data['timestamp']).dt.days
    # train_data = train_data.drop_duplicates(subset=['user_id', 'item_id', 'day'], keep='last') # 31788324 -> 28575395 rows
    # train_data = train_data.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])
    # train_data = train_data.reset_index(drop=True)

    logger.info('Split train / test.')
    target_columns = ['user_id', 'item_id', 'price', 'timestamp', 'day', 'count', 'occurence', 'product_code', 'product_type_no', 'graphical_appearance_no',
                      'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_group_no', 'section_no', 'garment_group_no']
    train_target_df = train_data[target_columns]
    test_target_df = train_data[target_columns] # 동일한 purchase_user_item.csv를 쓰므로 train_target_df와 동일하게 설정
    
    # 2. target_day 이전 날짜와 target_day 해당 날짜의 데이터를 분리해 각각 train_df, test_df에 저장
    if config['recent_two_weeks']:
        train_df = train_target_df[(train_target_df['day'] < target_day) & (train_target_df['day'] > (target_day - 15))] # 28575395 -> 534352 rows
    else:
        train_df = train_target_df[train_target_df['day'] < target_day]
    test_df = test_target_df[test_target_df['day'] == target_day] # 53280 rows

    # 3. train_df를 count_high와 count_low 사이값으로 자름
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
    unique_last_test_df = last_test_df.drop_duplicates(['user_id']) # 53280 rows -> 16092 rows, user별 last_order에서 user_id가 (장바구니에서)중복되는 것을 제거.

    # 5. dtype을 category로 변경
    # 5-1. train_df
    train_df['user_id'] = train_df['user_id'].astype('category')
    train_df['item_id'] = train_df['item_id'].astype('category')
    # train_df['age'] = train_df['age'].astype('int64')
    # train_df['price'] = train_df['price'].astype('int64')
    # train_df['postal_code'] = train_df['postal_code'].astype('category')
    train_df['product_code'] = train_df['product_code'].astype('category')
    train_df['product_type_no'] = train_df['product_type_no'].astype('category')
    train_df['graphical_appearance_no'] = train_df['graphical_appearance_no'].astype('category')
    train_df['colour_group_code'] = train_df['colour_group_code'].astype('category')
    train_df['perceived_colour_value_id'] = train_df['perceived_colour_value_id'].astype('category')
    train_df['perceived_colour_master_id'] = train_df['perceived_colour_master_id'].astype('category')
    train_df['department_no'] = train_df['department_no'].astype('category')
    train_df['index_group_no'] = train_df['index_group_no'].astype('category')
    train_df['section_no'] = train_df['section_no'].astype('category')
    train_df['garment_group_no'] = train_df['garment_group_no'].astype('category')
    
    # 5-2. test_df
    test_df['user_id'] = test_df['user_id'].astype('category')
    test_df['item_id'] = test_df['item_id'].astype('category')
    # test_df['age'] = test_df['age'].astype('int64')
    # test_df['price'] = test_df['price'].astype('int64')
    # test_df['postal_code'] = test_df['postal_code'].astype('category')
    #test_df['product_code'] = test_df['product_code'].astype('category')
    #test_df['product_type_no'] = test_df['product_type_no'].astype('category')
    #test_df['graphical_appearance_no'] = test_df['graphical_appearance_no'].astype('category')
    #test_df['colour_group_code'] = test_df['colour_group_code'].astype('category')
    #test_df['perceived_colour_value_id'] = test_df['perceived_colour_value_id'].astype('category')
    #test_df['perceived_colour_master_id'] = test_df['perceived_colour_master_id'].astype('category')
    #test_df['department_no'] = test_df['department_no'].astype('category')
    #test_df['index_group_no'] = test_df['index_group_no'].astype('category')
    #test_df['section_no'] = test_df['section_no'].astype('category')
    #test_df['garment_group_no'] = test_df['garment_group_no'].astype('category')
    
    # 5-3. unique_last_test_df
    unique_last_test_df['user_id'] = unique_last_test_df['user_id'].astype('category')
    unique_last_test_df['item_id'] = unique_last_test_df['item_id'].astype('category')
    # unique_last_test_df['age'] = unique_last_test_df['age'].astype('int64')
    # unique_last_test_df['price'] = unique_last_test_df['price'].astype('int64')
    # unique_last_test_df['postal_code'] = unique_last_test_df['postal_code'].astype('category')
    #unique_last_test_df['product_code'] = unique_last_test_df['product_code'].astype('category')
    #unique_last_test_df['product_type_no'] = unique_last_test_df['product_type_no'].astype('category')
    #unique_last_test_df['graphical_appearance_no'] = unique_last_test_df['graphical_appearance_no'].astype('category')
    #unique_last_test_df['colour_group_code'] = unique_last_test_df['colour_group_code'].astype('category')
    #unique_last_test_df['perceived_colour_value_id'] = unique_last_test_df['perceived_colour_value_id'].astype('category')
    #unique_last_test_df['perceived_colour_master_id'] = unique_last_test_df['perceived_colour_master_id'].astype('category')
    #unique_last_test_df['department_no'] = unique_last_test_df['department_no'].astype('category')
    #unique_last_test_df['index_group_no'] = unique_last_test_df['index_group_no'].astype('category')
    #unique_last_test_df['section_no'] = unique_last_test_df['section_no'].astype('category')
    #unique_last_test_df['garment_group_no'] = unique_last_test_df['garment_group_no'].astype('category')
    
    combined_user_ids = copy.deepcopy(train_df['user_id']).append(unique_last_test_df['user_id'], ignore_index=True).astype('category')
    combined_item_ids = copy.deepcopy(train_df['item_id']).append(unique_last_test_df['item_id'], ignore_index=True).astype('category')
    combined_prodct_code_ids = copy.deepcopy(train_df['product_code']).append(unique_last_test_df['product_code'], ignore_index=True).astype('category')
    combined_prodct_type_ids = copy.deepcopy(train_df['product_type_no']).append(unique_last_test_df['product_type_no'], ignore_index=True).astype('category')
    combined_graphical_appearance_ids = copy.deepcopy(train_df['graphical_appearance_no']).append(unique_last_test_df['graphical_appearance_no'], ignore_index=True).astype('category')
    combined_colour_group_code_ids = copy.deepcopy(train_df['colour_group_code']).append(unique_last_test_df['colour_group_code'], ignore_index=True).astype('category')
    combined_perceived_colour_value_ids = copy.deepcopy(train_df['perceived_colour_value_id']).append(unique_last_test_df['perceived_colour_value_id'], ignore_index=True).astype('category')
    combined_perceived_colour_master_ids = copy.deepcopy(train_df['perceived_colour_master_id']).append(unique_last_test_df['perceived_colour_master_id'], ignore_index=True).astype('category')
    combined_department_ids = copy.deepcopy(train_df['department_no']).append(unique_last_test_df['department_no'], ignore_index=True).astype('category')
    combined_index_group_ids = copy.deepcopy(train_df['index_group_no']).append(unique_last_test_df['index_group_no'], ignore_index=True).astype('category')
    combined_section_ids = copy.deepcopy(train_df['section_no']).append(unique_last_test_df['section_no'], ignore_index=True).astype('category')
    combined_garment_group_ids = copy.deepcopy(train_df['garment_group_no']).append(unique_last_test_df['garment_group_no'], ignore_index=True).astype('category')
    
    # 6. user/item dictionary 생성, interaction과 mapping
    logger.info('Make user item index dictionary & map to interactions.')
    
    # 6-1. user, item, item_meta idx mapping set 생성
    train_user2idx = {user_id: idx for idx, user_id in enumerate(train_df['user_id'].cat.categories)}
    train_item2idx = {item_id: idx for idx, item_id in enumerate(train_df['item_id'].cat.categories)}
    train_product_code2idx = {product_code: idx for idx, product_code in enumerate(train_df['product_code'].cat.categories)}
    train_product_type2idx = {product_type: idx for idx, product_type in enumerate(train_df['product_type_no'].cat.categories)}
    train_graphical_appearance2idx = {graphical_appearance: idx for idx, graphical_appearance in enumerate(train_df['graphical_appearance_no'].cat.categories)}
    train_colour_group_code2idx = {colour_group_code: idx for idx, colour_group_code in enumerate(train_df['colour_group_code'].cat.categories)}
    train_perceived_colour_value2idx = {perceived_colour_value: idx for idx, perceived_colour_value in enumerate(train_df['perceived_colour_value_id'].cat.categories)}
    train_perceived_colour_master2idx = {perceived_colour_master: idx for idx, perceived_colour_master in enumerate(train_df['perceived_colour_master_id'].cat.categories)}
    train_department2idx = {department_no: idx for idx, department_no in enumerate(train_df['department_no'].cat.categories)}
    train_index_group2idx = {index_group_no: idx for idx, index_group_no in enumerate(train_df['index_group_no'].cat.categories)}
    train_section2idx = {section_no: idx for idx, section_no in enumerate(train_df['section_no'].cat.categories)}
    train_garment_group2idx = {garment_group_no: idx for idx, garment_group_no in enumerate(train_df['garment_group_no'].cat.categories)}

    user2idx = {user_id: idx for idx, user_id in enumerate(combined_user_ids.cat.categories)}
    idx2user = {idx: user_id for user_id, idx in user2idx.items()}
    item2idx = {item_id: idx for idx, item_id in enumerate(combined_item_ids.cat.categories)}
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}
    product_code2idx = {product_code: idx for idx, product_code in enumerate(combined_prodct_code_ids.cat.categories)}
    idx2product_code = {idx: product_code for product_code, idx in product_code2idx.items()}
    product_type2idx = {product_type: idx for idx, product_type in enumerate(combined_prodct_type_ids.cat.categories)}
    idx2product_type = {idx: product_type for product_type, idx in product_type2idx.items()}
    graphical_appearance2idx = {graphical_appearance: idx for idx, graphical_appearance in enumerate(combined_graphical_appearance_ids.cat.categories)}
    idx2graphical_appearance = {idx: graphical_appearance for graphical_appearance, idx in graphical_appearance2idx.items()}
    colour_group2idx = {colour_group: idx for idx, colour_group in enumerate(combined_colour_group_code_ids.cat.categories)}
    idx2colour_group = {idx: colour_group for colour_group, idx in colour_group2idx.items()}
    perceived_colour_value2idx = {perceived_colour_value: idx for idx, perceived_colour_value in enumerate(combined_perceived_colour_value_ids.cat.categories)}
    idx2perceived_colour_value = {idx: perceived_colour_value for perceived_colour_value, idx in perceived_colour_value2idx.items()}
    perceived_colour_master2idx = {perceived_colour_master: idx for idx, perceived_colour_master in enumerate(combined_perceived_colour_master_ids.cat.categories)}
    idx2perceived_colour_master = {idx: perceived_colour_master for idx, perceived_colour_master in perceived_colour_master2idx.items()}
    department2idx = {department: idx for idx, department in enumerate(combined_department_ids.cat.categories)}
    idx2department = {idx: department for department, idx in department2idx.items()}
    index_group2idx = {index_group: idx for idx, index_group in enumerate(combined_index_group_ids.cat.categories)}
    idx2index_group = {idx: index_group for index_group, idx in index_group2idx.items()}
    section2idx = {section: idx for idx, section in enumerate(combined_section_ids.cat.categories)}
    idx2section = {idx: section for section, idx in section2idx.items()}
    garment_group2idx = {garment_group: idx for idx, garment_group in enumerate(combined_garment_group_ids.cat.categories)}
    idx2garment_group = {idx: garment_group for garment_group, idx in garment_group2idx.items()}
    
    # 6-2. user_id, item_id, item_meta를 각각 user_idx, item_idx, meta_idx로 매핑
    train_df['user_id'] = train_df['user_id'].apply(lambda x: user2idx[x])
    train_df['item_id'] = train_df['item_id'].apply(lambda x: item2idx[x])
    train_df['product_code'] = train_df['product_code'].apply(lambda x: product_code2idx[x])
    train_df['product_type_no'] = train_df['product_type_no'].apply(lambda x: product_type2idx[x])
    train_df['graphical_appearance_no'] = train_df['graphical_appearance_no'].apply(lambda x: graphical_appearance2idx[x])
    train_df['colour_group_code'] = train_df['colour_group_code'].apply(lambda x: colour_group2idx[x])
    train_df['perceived_colour_value_id'] = train_df['perceived_colour_value_id'].apply(lambda x: perceived_colour_value2idx[x])
    train_df['perceived_colour_master_id'] = train_df['perceived_colour_master_id'].apply(lambda x: perceived_colour_master2idx[x])
    train_df['department_no'] = train_df['department_no'].apply(lambda x: department2idx[x])
    train_df['index_group_no'] = train_df['index_group_no'].apply(lambda x: index_group2idx[x])
    train_df['section_no'] = train_df['section_no'].apply(lambda x: section2idx[x])
    train_df['garment_group_no'] = train_df['garment_group_no'].apply(lambda x: garment_group2idx[x])    
    
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
    
    num_user = len(user2idx)
    num_item = len(item2idx)
    num_product_code = len(product_code2idx)
    num_product_type = len(product_type2idx)
    num_graphical_appearance = len(graphical_appearance2idx)
    num_colour_group = len(colour_group2idx)
    num_perceived_colour_value = len(perceived_colour_value2idx)
    num_perceived_colour_master = len(perceived_colour_master2idx)
    num_department = len(department2idx)
    num_index_group = len(index_group2idx)
    num_section = len(section2idx)
    num_garment_group = len(garment_group2idx)
    
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
    data_dict['num_product_code'] = num_product_code
    data_dict['num_product_type'] = num_product_type
    data_dict['num_graphical_appearance'] = num_graphical_appearance
    data_dict['num_colour_group'] = num_colour_group
    data_dict['num_perceived_colour_value'] = num_perceived_colour_value
    data_dict['num_perceived_colour_master'] = num_perceived_colour_master
    data_dict['num_department'] = num_department
    data_dict['num_index_group'] = num_index_group
    data_dict['num_section'] = num_section
    data_dict['num_garment_group'] = num_garment_group
    
    data_dict['user2idx'] = user2idx
    data_dict['idx2user'] = idx2user
    data_dict['item2idx'] = item2idx
    data_dict['idx2item'] = idx2item
    data_dict['product_code2idx'] = product_code2idx
    data_dict['idx2product_code'] = idx2product_code
    data_dict['product_type2idx'] = product_type2idx
    data_dict['idx2product_type'] = idx2product_type
    data_dict['graphical_appearance2idx'] = graphical_appearance2idx
    data_dict['idx2graphical_appearance'] = idx2graphical_appearance
    data_dict['colour_group2idx'] = colour_group2idx
    data_dict['idx2_colour_group'] = idx2colour_group
    data_dict['perceived_colour_value'] = perceived_colour_value2idx
    data_dict['idx2perceived_colour_value'] = idx2perceived_colour_value
    data_dict['perceived_colour_master'] = perceived_colour_master2idx
    data_dict['idx2perceived_colour_master'] = idx2perceived_colour_master
    data_dict['department2idx'] = department2idx
    data_dict['idx2department'] = idx2department
    data_dict['index_group2idx'] = index_group2idx
    data_dict['idx2index_group'] = idx2index_group
    data_dict['section2idx'] = section2idx
    data_dict['idx2section'] = idx2section
    data_dict['garment_group2idx'] = garment_group2idx
    data_dict['idx2garment_group'] = idx2garment_group
    
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
    hm_prep_meta(
        config
    ) 