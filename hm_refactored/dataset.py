from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset    
import random
import time

class NegativeSampler:
    def __init__(self, data_dict):
        self.item_cand = {}
        self.num_item = data_dict['num_item']
        self.train_dict = data_dict['user_train_dict'] # {user_id: [item_id1, item_id2, ...]} ( {0: [1, 2], 1: [3, 4, 5], ...} )
    
    def sampling(self, batch): # batch가 어떤 형태로 되어있는가? -> train.py에서 넣어주는거 보니깐 data_dict 형태인듯
        triplets = []
        for pos_pair in batch: # pos_pair: key-value pair
            neg = np.random.randint(self.num_item) # num_item 개수만큼 num_item range에서 random int의 1-dim matrix 생성
            while neg in self.train_dict[pos_pair[0]]: # pos_pair[0]: key-value pair에서 key?
                neg = np.random.randint(self.num_item) # user_id마다 neg 초기화
                break
            triplet = np.expand_dims(np.concatenate([pos_pair, [neg]], axis=0), 0) # [[0: [1,2], [neg1, neg2, ... ]]] 형태인지?
            triplets.append(triplet)
        triplets = np.concatenate(triplets, axis=0) # triplet의 array
        return triplets

class MultipleNegativeSampler:
    def __init__(self, data_dict, num_negatives=100):
        self.num_item = data_dict['num_item']
        self.train_dict = data_dict['user_train_dict']

        self.num_negatives = num_negatives
    
    def sampling(self, batch):
        pos_negs_pairs = []
        for pos_pair in batch:
            if int(pos_pair[1]) != self.num_item - 1: # pos_pair[1]: key-value pair에서 value?, int(pos_pair[1])는 value list의 length?
                negs = np.array([*random.sample(range(0, int(pos_pair[1])), int(pos_pair[1] / self.num_item * self.num_negatives)), 
                                *random.sample(range(int(pos_pair[1])+1, self.num_item), self.num_negatives - int(pos_pair[1] / self.num_item * self.num_negatives))]).astype(pos_pair.dtype)
            else:
                negs = np.array(random.sample(range(self.num_item - 1), self.num_negatives)).astype(pos_pair.dtype)
            pos_negs = np.expand_dims(np.concatenate([pos_pair, negs], axis=-1), axis=0)
            pos_negs_pairs.append(pos_negs)
        pos_negs_pairs = np.concatenate(pos_negs_pairs, axis=0)
        return pos_negs_pairs

class TwoViewSampler:
    def __init__(self, data_dict, seq_len):
        self.item_cand = {}
        self.num_user = data_dict['num_user']
        self.num_item = data_dict['num_item']
        self.user_train_dict = data_dict['user_train_dict']
        self.item_train_dict = data_dict['item_train_dict']

        self.users = data_dict['train_df'][['user_id', 'item_id', 'occurence']].values.astype(np.float32)
        # self.users = data_dict['train_df']['user_id'].unique().to_numpy().astype(np.int32)

        self.seq_len = seq_len
    
    def sampling(self, batch):
        two_view_samples = []
        for pos_pair in batch:
            neg = np.random.randint(self.num_item)
            while neg in self.user_train_dict[pos_pair[0]]:
                neg = np.random.randint(self.num_item)

            random_idx = np.random.randint(self.users.shape[0])
            neg_user = self.users[random_idx][0]
            while neg_user in self.item_train_dict[pos_pair[1]]:
                random_idx = np.random.randint(self.users.shape[0])
                neg_user = self.users[random_idx][0]

            neg_user_interaction = self.user_train_dict.get(int(neg_user))

            neg_user_history, neg_user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
            if len(neg_user_interaction) > 0:
                occurence = int(self.users[random_idx][2])
                valid_history = neg_user_interaction[:occurence]
                valid_history = valid_history[-1 * self.seq_len:]
                starting_idx = -1 * len(valid_history) if len(valid_history) > 0 else len(neg_user_history)
                neg_user_history[starting_idx:] = valid_history
                neg_user_history_mask[starting_idx:] = np.ones_like(valid_history) 

            # neg_user = np.random.choice(np.setdiff1d(self.users, self.item_train_dict[pos_pair[1]]))
            # neg_user_interaction = self.user_train_dict.get(neg_user)

            # start = random.sample(range(len(neg_user_interaction)), 1)[0]
            # end = random.sample(range(start + 1, len(neg_user_interaction) + 1), 1)[0]

            two_view_sample = np.expand_dims(np.concatenate([pos_pair, neg_user_history, neg_user_history_mask, [neg]], axis=0), 0)
            two_view_samples.append(two_view_sample)          

        two_view_samples = np.concatenate(two_view_samples, axis=0)
        return two_view_samples

class TwoViewConsistentSampler:
    def __init__(self, data_dict, seq_len):
        self.item_cand = {}
        self.num_user = data_dict['num_user']
        self.num_item = data_dict['num_item']
        self.user_train_dict = data_dict['user_train_dict']
        self.item_train_dict = data_dict['item_train_dict']

        self.users = data_dict['train_df'][['user_id', 'item_id', 'occurence']].values.astype(np.float32)

        self.seq_len = seq_len
    
    def sampling(self, batch):
        two_view_samples = []
        for pos_pair in batch:
            random_idx = np.random.randint(self.users.shape[0])
            neg_user = self.users[random_idx][0]
            while neg_user in self.item_train_dict[pos_pair[1]]:
                random_idx = np.random.randint(self.users.shape[0])
                neg_user = self.users[random_idx][0]

            neg_user_interaction = self.user_train_dict.get(int(neg_user))

            neg_user_history, neg_user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
            if len(neg_user_interaction) > 0:
                occurence = int(self.users[random_idx][2])
                valid_history = neg_user_interaction[:occurence]
                valid_history = valid_history[-1 * self.seq_len:]
                starting_idx = -1 * len(valid_history) if len(valid_history) > 0 else len(neg_user_history)
                neg_user_history[starting_idx:] = valid_history
                neg_user_history_mask[starting_idx:] = np.ones_like(valid_history)

            two_view_sample = np.expand_dims(np.concatenate([pos_pair, neg_user_history, neg_user_history_mask, [int(self.users[random_idx][1])]], axis=0), 0)
            two_view_samples.append(two_view_sample)          

        two_view_samples = np.concatenate(two_view_samples, axis=0)
        return two_view_samples

class TwoViewRandomSampler:
    def __init__(self, data_dict, seq_len):
        self.item_cand = {}
        self.num_user = data_dict['num_user']
        self.num_item = data_dict['num_item']
        self.user_train_dict = data_dict['user_train_dict']
        self.item_train_dict = data_dict['item_train_dict']

        self.users = data_dict['train_df'][['user_id', 'item_id', 'occurence']].values.astype(np.float32)

        self.seq_len = seq_len
    
    def sampling(self, batch):
        two_view_samples = []
        for pos_pair in batch:
            random_idx = np.random.randint(self.users.shape[0])
            while int(self.users[random_idx][1]) == int(pos_pair[1]):
                random_idx = np.random.randint(self.users.shape[0])

            neg_user = self.users[random_idx][0]
            neg_user_interaction = self.user_train_dict.get(int(neg_user))

            neg_user_history, neg_user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
            if len(neg_user_interaction) > 0:
                occurence = int(self.users[random_idx][2])
                valid_history = neg_user_interaction[:occurence]
                valid_history = valid_history[-1 * self.seq_len:]
                starting_idx = -1 * len(valid_history) if len(valid_history) > 0 else len(neg_user_history)
                neg_user_history[starting_idx:] = valid_history
                neg_user_history_mask[starting_idx:] = np.ones_like(valid_history)

            two_view_sample = np.expand_dims(np.concatenate([pos_pair, neg_user_history, neg_user_history_mask, [int(self.users[random_idx][1])]], axis=0), 0)
            two_view_samples.append(two_view_sample)          

        two_view_samples = np.concatenate(two_view_samples, axis=0)
        return two_view_samples

class TrainDataset(Dataset):
    def __init__(self, data_dict, seq_len):
        super(TrainDataset, self).__init__()
        self.pos_pairs = data_dict['train_df'][['user_id', 'item_id']].values.astype(np.float32)
        self.train_dict = data_dict['user_train_dict']

        self.seq_len = int(seq_len)

        self.occurences = data_dict['train_df'][['occurence']].values.astype(np.int32)

    def __len__(self):
        return len(self.pos_pairs)
    
    def __getitem__(self, idx):
        # return self.pos_pairs[idx]
        
        pos_pair = self.pos_pairs[idx] # [user_id, item_id]

        user_interaction = self.train_dict.get(int(pos_pair[0]), []) # user_id에 해당하는 item_id리스트 반환. 없으면 []반환. float32 type이므로 int32로 변환.

        # user_history, user_history_mask = np.zeros(self.seq_len), np.concatenate([[1], np.zeros(self.seq_len-1)])
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
        if len(user_interaction) > 0:
            # valid_history = user_interaction[:user_interaction.index(int(pos_pair[1]))]
            occurence = self.occurences[idx][0]
            valid_history = user_interaction[:occurence]
            valid_history = valid_history[-1 * self.seq_len:] # valid_history를 seq_len길이만큼 뒤에서부터 자름
            starting_idx = -1 * len(valid_history) if len(valid_history) > 0 else len(user_history)
            user_history[starting_idx:] = valid_history
            user_history_mask[starting_idx:] = np.ones_like(valid_history)
        train_dataset = np.concatenate([pos_pair, user_history, user_history_mask])
        return train_dataset

class TrainMetaDataset(Dataset):
    def __init__(self, data_dict, seq_len):
        super(TrainMetaDataset, self).__init__()
        self.pos_pairs_with_meta = data_dict['train_df'][['user_id', 'item_id', 'product_type_no', 'department_no', 'garment_group_no', 'age']].values.astype(np.float32)
        self.train_dict = data_dict['user_train_dict']

        self.seq_len = int(seq_len)

        self.occurences = data_dict['train_df'][['occurence']].values.astype(np.int32)

    def __len__(self):
        return len(self.pos_pairs_with_meta)
    
    def __getitem__(self, idx):
        # return self.pos_pairs[idx]
        
        pos_pair_with_meta = self.pos_pairs_with_meta[idx] # [user_id, item_id, product_code, product_type_no, graphical_appearance_no, colour_group_code,
                                                           #  perceived_colour_value_id, perceived_colour_master_id, department_no, index_group_no, section_no,
                                                           #  garment_group_no, age]
        
        user_interaction = self.train_dict.get(int(pos_pair_with_meta[0]), []) # user_id에 해당하는 item_id리스트 반환. 없으면 []반환. float32 type이므로 int32로 변환.

        # user_history, user_history_mask = np.zeros(self.seq_len), np.concatenate([[1], np.zeros(self.seq_len-1)])
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
        if len(user_interaction) > 0:
            occurence = self.occurences[idx][0]
            valid_history = user_interaction[:occurence]
            valid_history = valid_history[-1 * self.seq_len:] # valid_history를 seq_len길이만큼 뒤에서부터 자름
            starting_idx = -1 * len(valid_history) if len(valid_history) > 0 else len(user_history)
            user_history[starting_idx:] = valid_history
            user_history_mask[starting_idx:] = np.ones_like(valid_history)
        train_dataset = np.concatenate([pos_pair_with_meta, user_history, user_history_mask])
        return train_dataset

class TestDataset(Dataset):
    def __init__(self, data_dict, seq_len):
        super(TestDataset, self).__init__()
        self.train_dict = data_dict['user_train_dict']
        self.num_item = data_dict['num_item']
        self.user2idx = data_dict['user2idx']

        self.seq_len = int(seq_len)

        # test_df = data_dict['test_df'][['user_id']].drop_duplicates()
        test_df = data_dict['unique_last_test_df'][['user_id', 'item_id']]
        
        test_df['user_idx'] = test_df['user_id'].apply(lambda x: self.user2idx.get(x, -1))
        
        # self.test_data = test_df[test_df['user_idx'] != -1]

        self.test_data = test_df
        self.test_data['user_id'] = self.test_data['user_idx']
        self.test_data = self.test_data.drop('user_idx', axis=1)
        self.test_data = self.test_data.values.astype(np.float32)
        self.cold_users = test_df[test_df['user_idx'] == -1]['user_id'].unique()

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        u = self.test_data[idx] # user_idx, item_id pair
        user_train_data = self.train_dict.get(u[0], []) # user_idx에 해당하는 list of item_id
        mask = np.ones(self.num_item)
        mask[user_train_data] = -10000
        
        #print("idx:", idx)
        #print("self.test_data:", self.test_data)
        #print("u:", u)
        #print("user_train_data", user_train_data)
        #print("mask", mask)
        # return u, mask

        user_interaction = self.train_dict.get(int(u[0]), [])
        
        # user_history, user_history_mask = np.zeros(self.seq_len), np.concatenate([[1], np.zeros(self.seq_len-1)])
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
        if len(user_interaction) > 0:
            user_interaction = user_interaction[-1 * self.seq_len:]
            starting_idx = -1 * len(user_interaction)
            user_history[starting_idx:] = user_interaction
            user_history_mask[starting_idx:] = np.ones_like(user_interaction)
        #print("user_history:", user_history, "user_history_mask:", user_history_mask)
        return u, mask, user_history, user_history_mask

class BenchmarkDataset(Dataset):
    def __init__(self, data_dict, seq_len, num_negatives=100):
        super(BenchmarkDataset, self).__init__()
        self.train_dict = data_dict['user_train_dict']
        self.num_item = data_dict['num_item']
        self.user2idx = data_dict['user2idx']
        self.item2idx = data_dict['item2idx']
        self.idx2item = data_dict['idx2item']

        self.seq_len = int(seq_len)

        test_df = data_dict['unique_last_test_df'][['user_id', 'item_id']]
        
        test_df['user_idx'] = test_df['user_id'].apply(lambda x: self.user2idx.get(x, -1))
        test_df['item_idx'] = test_df['item_id'].apply(lambda x: self.item2idx.get(x, -1))
        
        # self.test_data = test_df[test_df['user_idx'] != -1]

        self.test_data = test_df
        self.test_data['user_id'] = self.test_data['user_idx']
        self.test_data['item_id'] = self.test_data['item_idx']

        unique_test_items = self.test_data['item_id'].unique().to_numpy()
        self.testitem2idx = {item_id: idx for idx, item_id in enumerate(unique_test_items)}
        self.testidx2item = {idx: item_id for item_id, idx in self.testitem2idx.items()}
        num_unique_item = len(self.testitem2idx)

        print("Num Unique Item : {}".format(num_unique_item))
        print("Num Negatives : {}".format(num_negatives))

        self.test_data = self.test_data.drop(['user_idx', 'item_idx'], axis=1)
        self.test_data = self.test_data.values.astype(np.float32)
        
        self.cold_users = test_df[test_df['user_idx'] == -1]['user_id'].unique()
        self.cold_items = test_df[test_df['item_idx'] == -1]['item_id'].unique()

        # batch_selecter = np.tile(np.arange(num_unique_item)[None,:], (num_unique_item, 1))[~np.eye(num_unique_item, dtype=bool)].reshape(num_unique_item, -1)
        # self.negatives = np.apply_along_axis(np.random.choice, axis=1, arr=batch_selecter, size=num_negatives, replace=False).astype(np.float32)

        sample_negatives = lambda x : np.random.choice(np.arange(num_unique_item)[:x].tolist() + np.arange(num_unique_item)[x+1:].tolist(), (num_negatives, ), replace=False)
        self.negatives = np.stack([sample_negatives(idx) for idx in range(num_unique_item)])

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        idx_to_items = np.array([self.testidx2item.get(int(i)) for i in self.negatives[self.testitem2idx[int(self.test_data[idx][-1])]]])
        # idx_to_items = np.array([self.idx2item.get(int(i)) for i in self.negatives[self.item2idx[int(self.test_data[idx][-1])]]])

        user_interaction = self.train_dict.get(int(self.test_data[idx][0]), [])

        # user_history, user_history_mask = np.zeros(self.seq_len), np.concatenate([[1], np.zeros(self.seq_len-1)])
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
        if len(user_interaction) > 0:
            user_interaction = user_interaction[-1 * self.seq_len:]
            starting_idx = -1 * len(user_interaction)
            user_history[starting_idx:] = user_interaction
            user_history_mask[starting_idx:] = np.ones_like(user_interaction)

        # return (np.concatenate([self.test_data[idx], self.negatives[self.testitem2idx[int(self.test_data[idx][-1])]]]), 
        #         np.concatenate([np.array([int(self.test_data[idx][-1])]), idx_to_items]), 
        #         user_history, user_history_mask)

        return (np.concatenate([self.test_data[idx], idx_to_items]), user_history, user_history_mask)

class BenchmarkOverAllDataset(Dataset):
    def __init__(self, data_dict, seq_len, num_negatives=100):
        super(BenchmarkOverAllDataset, self).__init__()
        self.train_dict = data_dict['user_train_dict']
        self.num_item = data_dict['num_item']
        self.user2idx = data_dict['user2idx']
        self.item2idx = data_dict['item2idx']
        self.idx2item = data_dict['idx2item']

        self.seq_len = int(seq_len)

        test_df = data_dict['unique_last_test_df'][['user_id', 'item_id']]
        
        test_df['user_idx'] = test_df['user_id'].apply(lambda x: self.user2idx.get(x, -1))
        test_df['item_idx'] = test_df['item_id'].apply(lambda x: self.item2idx.get(x, -1))

        self.test_data = test_df
        self.test_data['user_id'] = self.test_data['user_idx']
        self.test_data['item_id'] = self.test_data['item_idx']

        unique_test_items = self.test_data['item_id'].unique().to_numpy()
        self.origidx2newidx = {item_id: idx for idx, item_id in enumerate(unique_test_items)}
        self.newidx2origidx = {idx: item_id for item_id, idx in self.origidx2newidx.items()}
        num_unique_item = len(self.origidx2newidx)

        print("Num Unique Item : {}".format(num_unique_item))
        print("Num Negatives : {}".format(num_negatives))

        self.test_data = self.test_data.drop(['user_idx', 'item_idx'], axis=1)
        self.test_data = self.test_data.values.astype(np.float32)
        
        self.cold_users = test_df[test_df['user_idx'] == -1]['user_id'].unique()
        self.cold_items = test_df[test_df['item_idx'] == -1]['item_id'].unique()

        sample_negatives = lambda x : np.random.choice(np.arange(self.num_item)[:x].tolist() + np.arange(self.num_item)[x+1:].tolist(), (num_negatives, ), replace=False)
        self.negatives = np.stack([sample_negatives(idx) for idx in range(num_unique_item)])
        
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        idx_to_items = self.negatives[self.origidx2newidx[int(self.test_data[idx][-1])]]

        user_interaction = self.train_dict.get(int(self.test_data[idx][0]), [])

        # user_history, user_history_mask = np.zeros(self.seq_len), np.concatenate([[1], np.zeros(self.seq_len-1)])
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
        if len(user_interaction) > 0:
            user_interaction = user_interaction[-1 * self.seq_len:]
            starting_idx = -1 * len(user_interaction)
            user_history[starting_idx:] = user_interaction
            user_history_mask[starting_idx:] = np.ones_like(user_interaction)

        return (np.concatenate([self.test_data[idx], idx_to_items]), user_history, user_history_mask)

class BenchmarkTotalDataset(Dataset):
    def __init__(self, data_dict, seq_len, num_negatives=100):
        super(BenchmarkTotalDataset, self).__init__()
        self.train_dict = data_dict['user_train_dict']
        self.num_item = data_dict['num_item']
        self.user2idx = data_dict['user2idx']
        self.item2idx = data_dict['item2idx']
        self.idx2item = data_dict['idx2item']

        self.seq_len = int(seq_len)
        
        # target_day last_order data의 [user_id, item_id] pair
        test_df = data_dict['unique_last_test_df'][['user_id', 'item_id']]
        #print("test_df:", test_df)
        
        # user_id, item_id를 각각 user_idx, item_idx로 변환
        test_df['user_idx'] = test_df['user_id'].apply(lambda x: self.user2idx.get(x, -1)) # dict.get(x, -1): key가 x인 value를 return하고, 없으면 -1 return
        test_df['item_idx'] = test_df['item_id'].apply(lambda x: self.item2idx.get(x, -1))

        self.test_data = test_df
        self.test_data['user_id'] = self.test_data['user_idx'] # user_id column을 user_idx column으로 대체
        self.test_data['item_id'] = self.test_data['item_idx'] # item_id column을 item_idx column으로 대체

        self.unique_test_items = self.test_data['item_id'].unique().to_numpy() # item_idx의 고유한 값들만의 array
        num_unique_item = len(self.unique_test_items)
        #print("num_unique_item:", num_unique_item)

        print("Num Unique Item : {}".format(num_unique_item))
        print("Num Negatives : {}".format(num_negatives))

        self.test_data = self.test_data.drop(['user_idx', 'item_idx'], axis=1) # user_idx, item_idx column 제거
        self.test_data = self.test_data.values.astype(np.float32)
        #print("test_df after id to idx transition:", test_df)
        
        self.cold_users = test_df[test_df['user_idx'] == -1]['user_id'].unique()
        self.cold_items = test_df[test_df['item_idx'] == -1]['item_id'].unique()
        
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        negatives = np.array(list(set(self.unique_test_items).difference([int(self.test_data[idx][1])]))) # list(set())으로 list내 중복 제거. difference로 [item_idx]를 제거해 negative array생성
        
        user_interaction = self.train_dict.get(int(self.test_data[idx][0]), [])
        #print("self.unique_test_items:", self.unique_test_items)
        #print("self.test_data[idx]:", self.test_data[idx])
        #print("self.test_data[idx][0]:", self.test_data[idx][0], "self.test_data[idx][1]:", self.test_data[idx][1])
        #print("negatives:", negatives)
        #print("user_interaction:", user_interaction)
        #print("num_unique_item:", len(self.unique_test_items), "num_negatives:", len(negatives))

        # user_history, user_history_mask = np.zeros(self.seq_len), np.concatenate([[1], np.zeros(self.seq_len-1)])
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
        if len(user_interaction) > 0:
            user_interaction = user_interaction[-1 * self.seq_len:]
            starting_idx = -1 * len(user_interaction)
            user_history[starting_idx:] = user_interaction
            user_history_mask[starting_idx:] = np.ones_like(user_interaction)
        #print("user_history:", user_history, "user_history_mask:", user_history_mask)
        benchmark_dataset = (np.concatenate([self.test_data[idx], negatives]), user_history, user_history_mask)
        #print("benchmark_dataset:", benchmark_dataset)
        return benchmark_dataset 