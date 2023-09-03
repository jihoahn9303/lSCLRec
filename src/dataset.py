import pandas as pd
from typing import Set, Union, List 
import numpy as np 
import tensorflow as tf 
import math
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .augmentation import Substitution, Masking, Cropping
from .utils import transaction_item_idx


class IsimCLRDataLoader(Sequence):
    ''' class for contrastive learing custom dataloader '''
    def __init__(
          self, 
          user_indices: np.ndarray, 
          batch_size: int, 
          max_item_similarity_dict: dict, 
          item_sequence_dict: dict,
          sequence_len: int,
          substitute_rate: float = 0.3,
          mask_rate: float = 0.3,
          crop_rate: float = 0.3,
          data_mode: str = 'train',  # 'train', 'valid', 'test'
          aug_mode: str = 'substitute',  # 'substitute', 'crop', 'mask'
          n_views: int = 2,  # number of view
          shuffle=False,
    ):
        self.user_indices = user_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_mode = data_mode
        self.aug_mode = aug_mode  
        self.item_sequence_dict = item_sequence_dict
        self.sequence_len = sequence_len
        self.n_views = n_views

        self.max_item_similarity_dict = max_item_similarity_dict
        self.substitute_rate = substitute_rate
        self.masking_rate = mask_rate
        self.crop_rate = crop_rate
        self.substitution = Substitution(self.max_item_similarity_dict, self.substitute_rate, mode=self.data_mode)
        self.masking = Masking(self.masking_rate, mode=self.data_mode)
        self.cropping = Cropping(self.crop_rate, self.sequence_len, mode=self.data_mode)
        
        self.on_epoch_end()


    def __len__(self):
        return math.ceil(len(self.user_indices) / self.batch_size)

		
    def __getitem__(self, idx):
		# sampler
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = [user_idx for user_idx in indices]
        
        
        # get item list for each user in batch
        # using vectorize function for performance
        vectorize_func = np.vectorize(transaction_item_idx, otypes=[object])  
        item_sequence_for_user = vectorize_func(batch_x, self.item_sequence_dict)
        item_sequence_for_user = pad_sequences(
            item_sequence_for_user, maxlen=self.sequence_len, padding='post'
        )
       
        item_sequence: tf.Tensor

        if self.aug_mode == 'substitute':
            # substitue item
            item_sequence = [self.substitution(item_sequence_for_user) for _ in range(self.n_views)]

        elif self.aug_mode == 'crop':
            # crop item
            item_sequence = [self.masking(item_sequence_for_user) for _ in range(self.n_views)]

        elif self.aug_mode == 'mask':
            # mask item
            item_sequence = [self.masking(item_sequence_for_user) for _ in range(self.n_views)]

        return tf.convert_to_tensor(item_sequence, dtype=tf.int64)
                

    # shuffle user id for train dataset for each epoch
    def on_epoch_end(self):
        self.indices = self.user_indices
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
            
            
class RECDataLoader(Sequence):
    ''' class for recommendation system custom dataloader '''
    def __init__(
          self, 
          transaction_df: pd.DataFrame, 
          batch_size: int, 
          shuffle=False
    ):
        self.transaction_arr = transaction_df.to_numpy(dtype='int32')
        self.batch_size = batch_size
        self.shuffle = shuffle    
        self.on_epoch_end()


    def __len__(self):
        return math.ceil(self.transaction_arr.shape[0] / self.batch_size)

		
    def __getitem__(self, idx):
		# sampler
        batch_arr = self.transaction_arr[idx*self.batch_size:(idx+1)*self.batch_size, :]
        user_indices, item_incices, ratings = batch_arr[:, 0], batch_arr[:, 1], batch_arr[:, 2]
        user_indices = tf.convert_to_tensor(user_indices, dtype=tf.int32)
        item_incices = tf.convert_to_tensor(item_incices, dtype=tf.int32)
        ratings = tf.convert_to_tensor(ratings, dtype=tf.int32)          
        user_item_index_set = [user_indices, item_incices]
        
        return (user_item_index_set, ratings)
                

    # shuffle user id for train dataset for each epoch
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.transaction_arr)
            
            
            
class CLDiversityDataLoader(Sequence):
    ''' custom dataloader class for calculating diversity '''
    def __init__(
          self,
          user_id, 
          negative_item_indices: np.ndarray, 
          batch_size: int
    ):
        self.user_id = user_id
        self.negative_item_indices = negative_item_indices
        self.batch_size = batch_size


    def __len__(self):
        return math.ceil(self.negative_item_indices.shape[0] / self.batch_size)

		
    def __getitem__(self, idx):
		# sampler
        negative_batch_arr = self.negative_item_indices[idx*self.batch_size:(idx+1)*self.batch_size]
        user_batch_arr = np.ones(shape=negative_batch_arr.shape[0]) * self.user_id
        rating_batch_arr = np.zeros(shape=negative_batch_arr.shape[0])      
           
        return (user_batch_arr, negative_batch_arr, rating_batch_arr)
                
