import numpy as np 
import copy 
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .utils import (
    substitute_item_idx
)


# class for data substitution
class Substitution(object):
    """Substitute with similar items"""
    def __init__(
        self, 
        max_item_similarity_dict: dict, 
        substitute_rate: float,
        mode: str,
    ):    
        self.max_item_similarity_dict = max_item_similarity_dict
        self.substitute_rate = substitute_rate
        self.mode = mode  # 'train' or 'valid' or 'test'
        

    def __call__(
        self,   
        sequence: np.ndarray,
    ):
        '''
        sequence : numpy 2-D array  (# of users, # of items)
        '''
        assert (self.mode == 'train') or (self.mode == 'valid') or (self.mode == 'test')

        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        substitute_nums = int(self.substitute_rate * copied_sequence.shape[1])

        # sampling item list for substitution (return index)
        target_item_indices = np.random.choice(
            copied_sequence.shape[1], 
            size=substitute_nums,
            replace=False
        )     
        target_item_indices = np.unique(target_item_indices)
       
        # vectorize function for performance
        vectorize_func = np.vectorize(substitute_item_idx)

        # substitute target item for max similarity item
        copied_sequence[:, target_item_indices] = vectorize_func(
            copied_sequence[:, target_item_indices], 
            self.max_item_similarity_dict
        )

        return copied_sequence
    
    
# class for data masking
class Masking(object):
    def __init__(
        self, 
        masking_rate: float,
        mode: str,
    ):    
        self.masking_rate = masking_rate
        self.mode = mode  # 'train', 'valid', 'test'


    def __call__(self, sequence: np.ndarray):
        '''
        sequence : numpy 2-D array  (# of users, # of items)
        '''
        assert (self.mode == 'train') or (self.mode == 'valid') or (self.mode == 'test')

        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        masking_nums = int(self.masking_rate * copied_sequence.shape[1])

        # masking
        mask_idx = random.sample([i for i in range(copied_sequence.shape[1])], k=masking_nums)
        copied_sequence[:, mask_idx] = 0

        return copied_sequence
    
    
class Cropping(object):
    def __init__(
        self, 
        cropping_rate: float, 
        seq_len: int,
        mode: str,
    ):
        self.cropping_rate = cropping_rate
        self.seq_len = seq_len
        self.mode = mode

    def __call__(self, sequence: np.ndarray):
        '''
        sequence : numpy 2-D array  (# of users, # of items)
        '''
        assert (self.mode == 'train') or (self.mode == 'valid') or (self.mode == 'test')

        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        cropping_length = int(self.cropping_rate * len(copied_sequence.shape[1]))
         
        # cropping
        mask_idx = random.sample([i for i in range(copied_sequence.shape[1])], k=masking_nums)
        start_index = random.randint(0, len(copied_sequence[1]) - cropping_length - 1)
                   
        copied_sequence = copied_sequence[:, start_index:start_index+cropping_length]
        copied_sequence = pad_sequences(
            copied_sequence, 
            maxlen=self.seq_len, 
            padding='post'
        )

        return copied_sequence