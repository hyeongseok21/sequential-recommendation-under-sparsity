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

def gs_prep(config):
    
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
    # csv와 pkl로 구성되는 dataframe을 생성..?
    
    ### train_data 가공 ###
    train_data = load_funcs[train_data_type](train_data_path)
    train_data = train_data.fillna({'dealno' : 0.0}) # 'dealno' column의 결측값 처리
    
    if 'cart' in train_data_name:
        train_data = train_data[['pcid', 'prdid', 'dealno', 'siteid', 'sessionid', 'cartadddtm']] # 해당 column 선택
        train_data.columns = ['user_id', 'item_id', 'deal_no', 'site_id', 'session_id', 'timestamp'] # column명 변경 (pcid->user_id, prdid->item_id, cartadddtm->timestamp)
        train_data['timestamp'] = pd.to_datetime(train_data['timestamp']) # timestamp행의 dtype을 object에서 datetime64로 변경
    elif 'order' in train_data_name:
        train_data = train_data[['pcid', 'prdid', 'dealno', 'siteid', 'session', 'visitedtime']]
        train_data.columns = ['user_id', 'item_id', 'deal_no', 'site_id', 'session_id', 'timestamp']
        train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], unit='ms')
        
    train_num_days = (train_data['timestamp'].max() - train_data['timestamp'].min()).days # train_num_days 계산
    train_data['day'] = train_num_days - (train_data['timestamp'].max() - train_data['timestamp']).dt.days # train_data에 'day' column 추가
    train_data = train_data.drop_duplicates(subset=['user_id', 'item_id', 'day'], keep='last') # 중복 데이터 처리: subset내 column에서 'last'항목을 제외한 나머지 중복 데이터 제거. 15283350 rows -> 13499723 rows
    train_data = train_data.sort_values(by=['user_id', 'timestamp'], ascending=[True, True]) # user_id와 timestamp를 각각 오름차순으로 정렬
    train_data = train_data.reset_index(drop=True) # 정렬했으므로 index가 뒤죽박죽. 새로운 index를 추가하고(0, 1, 2, 3, ...), 기존 index는 column으로 insert한 뒤 제거

    ### test_data도 동일한 방식으로 가공 ###
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
        test_data = test_data.drop_duplicates(subset=['user_id', 'item_id', 'day'], keep='last') # 중복 데이터 처리: subset내 column에서 'last'항목을 제외한 나머지 중복 데이터를 True로 마킹
        test_data = test_data.sort_values(by=['user_id', 'timestamp'], ascending=[True, True]) # user_id와 timestamp를 각각 오름차순으로 정렬
        test_data = test_data.reset_index(drop=True) # 새로운 index를 추가하고(0, 1, 2, 3, ...), 기존 index는 column으로 insert한 뒤 제거

    logger.info('Split train / test.')
    target_columns = ['user_id', 'item_id', 'timestamp', 'day']
    train_target_df = train_data[target_columns]
    test_target_df = test_data[target_columns]
    ### train_data, test_data에서 target_columns의 column들만 선택해 train_target_df, test_target_df에 저장 ###
    

    # 2. target_day 이전 날짜와 target_day 해당 날짜의 데이터를 분리해 각각 train_df, test_df에 저장
    if config['recent_two_weeks']:
        train_df = train_target_df[(train_target_df['day'] < target_day) & (train_target_df['day'] > (target_day - 15))]
    else:
        train_df = train_target_df[train_target_df['day'] < target_day] # train_df: 13499723 rows -> 13499723 rows
    test_df = test_target_df[test_target_df['day'] == target_day] # test_df: 5788885 rows -> 177636 rows

    # 3. train_df에 count와 occurence column추가
    #if config['slice_user_by_count']:
    if True:
        tmp = train_df.groupby('user_id')['item_id'].count() # item id를 user_id별로 묶어 count
        print("tmp:", tmp)
        count = tmp.reset_index().rename(columns={"item_id": "count"}) # tmp의 index를 reset하고 item_id column명을 count로 명명
        print("count:", count)
        occurence = tmp.apply(lambda x : np.arange(x)).explode('item_id').rename("occurence") # item_id를 여러 행으로 전개. 0과 1로 표현 (마지막이 1)
        print("occurence:", occurence)
        
        train_df = train_df.merge(count, on='user_id', how='left')
        print("train_df after merge:", train_df)
        train_df = pd.concat([train_df, occurence], axis=1) # 위에서는 왜 merge를 했고 아래에는 왜 pd.concat을 했을지?
        print("train_df after concat:", train_df)
        
        # train_df = train_df[train_df['count'] < config['count_high']][train_df['count'] > config['count_low']]
        train_df = train_df[(train_df['count'] < config['count_high']) & (train_df['count'] > config['count_low'])] # count_low와 count_high 사이값을 가지는 row들만 선택


    # 4. train_df/test_df의 last_occur을 정의하고 last_test_df, unique_last_test_df 정의
    train_last_occur = (train_df.groupby(by='user_id')['timestamp'].transform(max) == train_df['timestamp']) # 해당 구매가 user의 last_order인지 아닌지 확인. boolean matrix
    last_train_df = train_df[train_last_occur] # 13499723 rows -> 1837593 rows
    test_last_occur = (test_df.groupby(by='user_id')['timestamp'].transform(max) == test_df['timestamp']) # boolean matrix. last timestamp인 것들만 걸러냄.
    last_test_df = test_df[test_last_occur] # 177636 rows -> 154983 rows. user별 last_order
    unique_last_test_df = last_test_df.drop_duplicates(['user_id']) # user별 last_order에서 user_id가 중복되는 것을 제거. 154983 rows -> 120156 rows. 장바구니에서 

    # 5. user_id/item_id dtype을 category로 변경
    train_df['user_id'] = train_df['user_id'].astype('category') # dtype: object->category, Categories (1832035, object) 1832035: # of unique_cart_user_id
    train_df['item_id'] = train_df['item_id'].astype('category') # dtype: int64->category, Categories (1214565, int64) 1214565: # of unique_cart_item_id
    test_df['user_id'] = test_df['user_id'].astype('category') # dtype: object->category, Categories (120156, object) 120156: # of unique_order_user_id
    test_df['item_id'] = test_df['item_id'].astype('category') # dtype: int64->category, Categories (60358, int64) 60358: # of unique_order_item_id
    unique_last_test_df['user_id'] = unique_last_test_df['user_id'].astype('category')
    # Categories (120156, object) 120156: # of unique_last_order_user_id -> test_df가 last_day + last_order로 구성되므로 test_df['user_id']의 category 수와 동일
    unique_last_test_df['item_id'] = unique_last_test_df['item_id'].astype('category')
    # Categories (41023, int64) 41023: # of unique_last_order_item_id -> 여러 user가 last_order로 같은 item을 선택했다는 뜻

    combined_user_ids = copy.deepcopy(train_df['user_id']).append(unique_last_test_df['user_id'], ignore_index=True).astype('category') # deepcopy: 리스트 내 객체들까지 모두 복사
    # Categories (1878937, object), Cart_user_id + Last_order_user_id
    combined_item_ids = copy.deepcopy(train_df['item_id']).append(unique_last_test_df['item_id'], ignore_index=True).astype('category')
    # Categories (1216502, int64), Cart_item_id + Last_order_item_id
    
    # 6. user/item dictionary 생성, interaction과 mapping
    logger.info('Make user item index dictionary & map to interactions.')
    # 6-1. user, item idx mapping set 생성
    train_user2idx = {user_id: idx for idx, user_id in enumerate(train_df['user_id'].cat.categories)} # {user_id: idx} set 구성. category수-1832035개
    train_item2idx = {item_id: idx for idx, item_id in enumerate(train_df['item_id'].cat.categories)} # {item_id: idx} set 구성. category수-1214565개

    user2idx = {user_id: idx for idx, user_id in enumerate(combined_user_ids.cat.categories)} # combined_user_ids로부터 {user_id: idx} set 구성. category수-1878937개
    idx2user = {idx: user_id for user_id, idx in user2idx.items()}
    item2idx = {item_id: idx for idx, item_id in enumerate(combined_item_ids.cat.categories)} # combined_item_ids로부터 {item_id: idx} set 구성. category수-1216502개
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}
    
    # 6-2. user_id, item_id를 각각 user_idx, item_idx로 매핑
    train_df['user_id'] = train_df['user_id'].apply(lambda x: user2idx[x]) # Length: 13499723, Categories (1832035, int64): [0, 1, 2, ..., 1878935, 1878936] ->?? 1832035 != 1878937
    train_df['item_id'] = train_df['item_id'].apply(lambda x: item2idx[x]) # Length: 13499723, Categories (1214565, int64): [0, 1, 2, ..., 1216500, 1216501] ->?? 1214565 != 1216502

    # 6-3. cold_user와 cold_item 제거
    if True:
    #if config['remove_cold_user']:
        test_df['cold_user_idx'] = test_df['user_id'].apply(lambda x: train_user2idx.get(x, -1)) # target_day 이전 선택이력이 없는 user 제거
        test_df = test_df[test_df['cold_user_idx'] != -1]

        unique_last_test_df['cold_user_idx'] = unique_last_test_df['user_id'].apply(lambda x: train_user2idx.get(x, -1))
        unique_last_test_df = unique_last_test_df[unique_last_test_df['cold_user_idx'] != -1]

    if True:
    #if config['remove_cold_item']:
        test_df['cold_item_idx'] = test_df['item_id'].apply(lambda x: train_item2idx.get(x, -1)) # target_day 이전 선택된 이력이 없는 item 제거
        test_df = test_df[test_df['cold_item_idx'] != -1]

        unique_last_test_df['cold_item_idx'] = unique_last_test_df['item_id'].apply(lambda x: train_item2idx.get(x, -1)) # user별 last_order에서도 target_day 이전 선택된 이력이 없는 item 제거
        unique_last_test_df = unique_last_test_df[unique_last_test_df['cold_item_idx'] != -1]

    num_item = len(item2idx) # slice_user_by_count: 1125954
    num_user = len(user2idx) # slice_user_by_count: 1105646

    # 6-4. train/test에 사용할 Dictionary 생성
    user_train_dict = train_df.sort_values(by='timestamp').groupby('user_id')['item_id'].apply(list).to_dict()
    # train_df(target_day 이전 cart data)를 timestamp별로 정렬 후, item_id를 user_id당 list로 묶어 dict형태로 저장. (e.g., {0: [1, 2], 1: [3, 4, 5], ...} )
    user_test_dict = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    # test_df(target_day의 order data)의 item_id를 user_id당 list로 묶어 dict형태로 저장. (e.g., {0: [1,2], 1: [3,4,5], ...} )
    user_last_test_dict = unique_last_test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    # unique_last_test_df(target_day의 order data중 user별 마지막 order)의 item_id를 user_id당 list로 묶어 dict형태로 저장
    item_train_dict = train_df.groupby('item_id')['user_id'].apply(list).to_dict()
    # train_df(target_day 이전 cart data)의 user_id를 item_id당 list로 묶어 dict형태로 저장
    
    import pdb; pdb.set_trace()
    
    # 7. zero history, recent_bought항목 제거
    if True:
    #if config['remove_zero_history']:
        idx = train_df.groupby(['user_id'])['timestamp'].transform(min) != train_df['timestamp'] # boolean matrix, train_df에서 user_id별로 timestamp가 최초인 것들을 False로 마킹
        train_df = train_df[idx] # idx vector를 사용해 train_df에서 timestamp이 최초가 아닌 것들만 선택
    
    if True:
    #if config['remove_recent_bought']:
        unique_last_test_df['recent_bought'] = unique_last_test_df.apply(lambda row: item2idx.get(row['item_id']) in user_train_dict[user2idx.get(row['user_id'])], axis=1) 
        unique_last_test_df = unique_last_test_df[unique_last_test_df['recent_bought'] == False] # recent_bought가 False인 것만 남김

    if True:
    #if config['remove_train_recent_bought']:
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
        default='user_item_cart.csv',
        help='File name of train data.'
    )
    parser.add_argument(
        '--test_data_name',
        type=str,
        default='user_item_order.csv',
        help='File name of test data.'
    )
    parser.add_argument(
        '--orig_path',
        type=str,
        default="/home/omnious/workspace/hyeongseok/datasets/gs",
        help='Folder path that original file saved.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default="/home/omnious/workspace/hyeongseok/datasets/gs/prep",
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
    gs_prep(
        config
    ) 