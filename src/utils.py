from typing import Set, List, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.math import l2_normalize
from tensorflow.keras.layers import Embedding, Layer
from einops import rearrange



def combine_representations(
    representation_1: tf.Tensor,
    representation_2: tf.Tensor
) -> tf.Tensor:
    '''
    combine two view tensors alternatively
    input:
        1) representation_1: 1st view (batch_size, # features * embebdding dim)
        2) representation_2: 2nd view (batch_size, # features * embebdding dim)
    ouput:
        combined tensor (2 * batch_size, # features * embebdding dim)
    '''
    return rearrange(
        [representation_1, representation_2], "n b d -> (b n) d"
    )
    
    
def generate_simclr_positive_indices(batch_size: int) -> tf.Tensor:
    '''
    gernerate indices for positive pair
    input:
        1) batch_size: batch size for training and testing
    ouput:
        alternated & combined tensor (2 * batch_size, )
        ex: if batch size == 4, this function will be return [1, 0, 3, 2, 5, 4, 7, 6]
    '''
    base = tf.range(batch_size)
    odds = base * 2 + 1
    even = base * 2

    return rearrange([odds, even], "n b -> (b n)")


def calculate_similarity(combined_representations: tf.Tensor) -> tf.Tensor:
    '''
    calculate similarity for each augmented pair
    input: 
        1) combined_representations (2 * batch size, # features * embedding dim)
    output: 
        similarity tensor for augmented pair (2 * batch size, 2 * batch size)
    '''
    # representations, _ = normalize(combined_representations, axis=1)  
    representations = l2_normalize(combined_representations, axis=1) 
    similarity = tf.einsum("id, jd -> ij", representations, representations)

    return similarity


def substitute_item_idx(item_id, max_sim_dict: dict):
    '''
    substitute source item with new item which has the highest similarity score
    input:
        1) max_sim_dict: dictionary that has most similar item index for each item
        {item index: most similar item index, ...}
        ex: {1: 3, 2: 5, 3: 1, 4: 10, ..., }
    '''
    if item_id == 0:
        return 0
    else:
        return max_sim_dict.get(item_id)
    
    
def transaction_item_idx(user_id, transaction_dict: dict):
    return transaction_dict.get(user_id)


def get_positive_sample(user_id, user_item_sequence_dict):
    '''
    get positive sample for each user
    input:
        1) user_id: target user id
        2) user_item_sequence_dict: dictionary that contains item indices rated by each user
           {user_id: item sequence, user_id: item sequence, ...}
           ex: {1: [3, 6, 1], 2: [10, 4, 15, 22]}
    output:
        positive sample tensor 
        ex: if user_id is 1, this function will be return [3, 6, 1] according to input example
    '''
    positive_sample = transaction_item_idx(user_id, user_item_sequence_dict)
    positive_sample = np.asarray(positive_sample, dtype=np.int32)
    
    return positive_sample


def get_negative_sample(user_id, user_item_sequence_dict, whole_item_list: np.ndarray):
    '''
    get positive sample for each user
    input:
        1) user_id: target user id
        2) user_item_sequence_dict: dictionary that contains item indices rated by each user
           {user_id: item sequence, user_id: item sequence, ...}
           ex: {1: [3, 6, 1], 2: [10, 4, 15, 22]}
        3) whole_item_list: array that contains the whole item indices
    output:
        negative sample tensor for target user
    '''
    positive_sample = get_positive_sample(user_id, user_item_sequence_dict)
    negative_sample = np.setdiff1d(whole_item_list, positive_sample) 
    
    return negative_sample


def get_item_category(item_id, category_dict: dict):
    if item_id == 0:
        return 0
    else:
        return category_dict.get(item_id)
    
    
def get_user_age(user_id, user_age_dict: dict):
    if user_id == 0:
        return 0
    else:
        return user_age_dict.get(user_id)
    
    
def get_user_height(user_id, user_height_dict: dict):
    if user_id == 0:
        return 0
    else:
        return user_height_dict.get(user_id)
    
    
def get_user_weight(user_id, user_weight_dict: dict):
    if user_id == 0:
        return 0
    else:
        return user_weight_dict.get(user_id)


def split_user_index(
    df: pd.DataFrame, 
    test_size: float,
    valid_size: float = None, 
    seed: int = 2023,
    stratify: str = 'is_vip',  # decided by transaction number for each user
) -> Set[np.ndarray]:
  
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[stratify],
        random_state=seed
    )

    real_train_df, valid_df = train_test_split(
        train_df, 
        test_size=valid_size, 
        stratify=train_df[stratify],
        random_state=seed
    )

        
    return (
        real_train_df.index.to_numpy(),  # train
        valid_df.index.to_numpy(), # validation
        test_df.index.to_numpy()   # test
    )
    
    
def select_user_by_sparsity(
    df: pd.DataFrame,
    test_user_index: np.ndarray,
    sparsity: float
) -> np.ndarray:
    target_df = df[df['user_id'].isin(test_user_index)]
    target_trans_df = calculate_transaction_num(target_df, type_str='user')
    threshold = target_trans_df['Transaction Count'].quantile(q=sparsity)
    final_series = target_trans_df[target_trans_df['Transaction Count'] <= threshold]
    final_user_index = final_series.index.to_numpy()
    
    return final_user_index 


def calculate_transaction_num(df: pd.DataFrame, type_str: str = 'user'):
    result_df: pd.DataFrame()
    
    if (type_str == 'item') or (type_str == 'user'):
        result_series = df.groupby(by=[type_str + "_id"])['rating'].count()
        indices = result_series.index
        result_df = pd.DataFrame(
            data={'Transaction Count': result_series.values}, 
            index=indices, 
        )
            
    else:
        result_df = None
    
    return result_df


def define_vip_user(trans_count: int, threshold: int) -> int:
    if trans_count >= threshold:
        return 1
    else:
        return 0


def make_lookup_table(
    info_dict: dict,
    dictionary_func, 
    data_type: str = 'int32'
):
    '''
    make lookup table for item's or user's ID
    lookup table: mapping user or item number to specific feature number
    input:
        1) info_dict: key -> Item's or user's ID, values: feature value that corresponds to ID
        2) data_type: return data type for feature value
        3) dictionary_func: function that returns feature value which corresponds to user's or item's id
    output:
        Lookup hash table Tensor(key: user's or item's ID -> value: feature value that corresponds to ID)
    '''
    # should be consistent with dictionary_func in ISimContrastiveLearning class and Recommender class
    ids = np.array(list(info_dict.keys()))
    vectorize_function = np.vectorize(dictionary_func, otypes=[data_type])
    features = vectorize_function(ids, info_dict)
    
    key_tensor = tf.constant(ids)
    key_tensor = tf.cast(key_tensor, dtype='int32')
    value_tensor = tf.constant(features)
    lookup_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(key_tensor, value_tensor),
        default_value=0
    )

    return lookup_table



def get_embbeding_layer(
    embedding_name: List[str],
    embedding_dim: int,
    embedding_size: List[int],
) -> List[Layer]:
    '''
    get embedding layer that corresponds to embedding_name (feature name)
    '''
    embedding_layer_list = list()

    for idx in range(len(embedding_name)):
        embedding_layer_list.append(
            Embedding(
              input_dim=embedding_size[idx] + 1, 
              output_dim=embedding_dim, 
              name=embedding_name[idx],            
            )
        )

    return embedding_layer_list


def make_padding_mask_tensor(arr: Union[np.ndarray, tf.Tensor]):
    '''
    make masking tensor from padded array
    input: arr -> shape: (# of user(batch), length of item sequence)
    output: Masking EagerTensor (if item id is 0, that value will be converted into False) 
           -> shape: (# of user, length of item sequence)
    '''
    return tf.not_equal(arr, 0)


def get_long_tail_item_list(trans_df: pd.DataFrame, k: float):
    item_transaction_count = calculate_transaction_num(trans_df, type_str='item')
    threshold = item_transaction_count['Transaction Count'].quantile(q=k)
    long_tail_item_list = item_transaction_count[item_transaction_count['Transaction Count'] < threshold].index.to_numpy()
    
    return long_tail_item_list



