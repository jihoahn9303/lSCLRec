import warnings 
warnings.filterwarnings("ignore")


import numpy as np
from src.utils import (
    get_negative_sample,
)
import tensorflow as tf
from src.dataset import CLDiversityDataLoader
from loguru import logger


class CLDiversity():
    '''
    Class for calculating diversity
    input:
        1) trained_model: trained model instance
        2) user_indices: indices of test users
        3) whole_test_items: indices of the whole test items
        4) long_tail_items: indices of long-tail item 
        5) user_item_sequence_dict: dictionary that contains item indices rated by each user
           {user_id: item sequence, user_id: item sequence, ...}
           ex: {1: [3, 6, 1], 2: [10, 4, 15, 22]}
    '''
    def __init__(
        self,
        trained_model,
        user_indices: np.ndarray,
        whole_test_items: np.ndarray, 
        long_tail_items: np.ndarray,
        user_item_sequence_dict: dict
    ):
        self.trained_model = trained_model
        self.trained_model.trainable = False
        
        self.user_indices = user_indices
        self.whole_test_items = whole_test_items
        self.long_tail_items = long_tail_items
        self.user_item_sequence_dict = user_item_sequence_dict    


    def predict(
        self,
        user_index,
    ) -> np.ndarray: 
        '''predict ratings of negative sample for target user'''
            
        ### negative sample id for specific user ###
        negative_sample = get_negative_sample(user_index, self.user_item_sequence_dict, self.whole_test_items)
        
        ### dataloader for calculating diversity ###
        diversity_dataloader = CLDiversityDataLoader(user_index, negative_sample, batch_size=512)

        ### predicted rating vector ###
        predicted_item_vector = np.zeros(shape=self.whole_test_items.shape[0])  
        
        for user_indices, negative_samples, ratings in diversity_dataloader:
            user_indices = tf.reshape(user_indices, shape=[user_indices.shape[0], -1])
            user_indices = tf.cast(user_indices, dtype='int32')
            negative_samples = tf.reshape(negative_samples, shape=[negative_samples.shape[0], -1])
            negative_samples = tf.cast(negative_samples, dtype='int32')
            ratings = tf.reshape(ratings, shape=[ratings.shape[0], -1])
            ratings = tf.cast(ratings, dtype='int32')
            inputs = tf.concat([user_indices, negative_samples, ratings], axis=-1)
            
            _, projection_head_output = self.trained_model(inputs, training=False)  # prediction
            
            predicted_ratings = projection_head_output.numpy()
            predicted_ratings = np.squeeze(predicted_ratings)
            negative_samples = negative_samples.numpy()
            negative_samples = np.squeeze(negative_samples) - 1
            
            predicted_item_vector.flat[negative_samples] = predicted_ratings
            
        return predicted_item_vector
        
    
    def get_lpc(
        self, 
        k: int
    ):
        lpc: np.float32
        set_for_recommended_total_item = set()
        set_for_recommended_long_tail_item = set()
        
        for idx, user_index in enumerate(self.user_indices):
            # prediction
            predicted_item_vector = self.predict(user_index)
            
            # get top k items
            sorted_index = np.argsort(predicted_item_vector)[::-1]
            top_k_sorted_index = sorted_index[:k] + 1
            set_for_recommended_total_item.update(top_k_sorted_index.tolist())
            
            # get predicted long-tail items
            predicted_long_tail_items = np.intersect1d(self.long_tail_items, top_k_sorted_index)
            set_for_recommended_long_tail_item.update(predicted_long_tail_items.tolist())
            
            # calculate lpc
            lpc = np.float32(len(set_for_recommended_long_tail_item) / len(set_for_recommended_total_item))
            
            # logging with loguru
            logger.info(
                f"Diversity (lpc): "
                f"K {k}\t"
                f"[{idx}/{self.user_indices.shape[0]}]\t"
                # f"lpc {sub_lpc:.6f}"
                f"lpc {lpc:.6f}"
            )
              
        return lpc


    def get_aplt(
        self,
        k: int
    ):
        aplt = np.float32(0.0)
        
        for idx, user_index in enumerate(self.user_indices):
            # prediction
            predicted_item_vector = self.predict(user_index)
            
            # get top k items   
            sorted_index = np.argsort(predicted_item_vector)[::-1]
            top_k_sorted_index = sorted_index[:k] + 1
            
            # get predicted long-tail items
            predicted_long_tail_items = np.intersect1d(self.long_tail_items, top_k_sorted_index)
        
            # calculate plt
            plt = predicted_long_tail_items.shape[0] / top_k_sorted_index.shape[0]
            aplt += np.float32(plt)
            
            # logging with loguru
            logger.info(
                f"Diversity (aplt): "
                f"K {k}\t"
                f"[{idx}/{self.user_indices.shape[0]}]\t"
                f"plt {plt:.6f}"
            )
            
        aplt /= np.float32(self.user_indices.shape[0])
        
        return aplt


    def get_aclt(
        self,
        k: int
    ):
        aclt = np.int32(0)
        
        for idx, user_index in enumerate(self.user_indices):
            # prediction
            predicted_item_vector = self.predict(user_index)
            
            # get top k items
            sorted_index = np.argsort(predicted_item_vector)[::-1]
            top_k_sorted_index = sorted_index[:k] + 1
            
            # get predicted long-tail items
            predicted_long_tail_items = np.intersect1d(self.long_tail_items, top_k_sorted_index)
        
            # calculate clt
            clt = predicted_long_tail_items.shape[0]
            aclt += clt 
            
            # logging with loguru
            logger.info(
                f"Diversity (aclt): "
                f"K {k}\t"
                f"[{idx}/{self.user_indices.shape[0]}]\t"
                f"clt {clt}"
            )
            
        aclt = np.float32(aclt)
        aclt /= np.float32(self.user_indices.shape[0])
        
        return aclt
        