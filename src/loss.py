import tensorflow as tf 
from tensorflow.nn import log_softmax
from tensorflow.linalg import set_diag, tensor_diag_part 
from tensorflow.experimental.numpy import finfo 
from tensorflow.math import divide_no_nan, reduce_mean, sqrt
from tensorflow.keras.losses import Loss, MeanSquaredError
from .utils import (
    combine_representations,
    calculate_similarity,
    generate_simclr_positive_indices
)


class NTXentLoss(Loss):
    ''' 
    Class for NT-Xent Loss
    input:
        1) temperature: temperature for NT-Xent Loss
    '''
    
    def __init__(self, temperature = 0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    

    def calculate_NTXentLoss(
        self,
        similarity_tensor: tf.Tensor, 
        positive_indicies: tf.Tensor
    ):     
        # shuffle columns axis by positive_indicies
        # each positive values will be placed in diagonal position
        similarity_tensor = tf.gather(similarity_tensor, indices=positive_indicies)
        positive_tensor = tensor_diag_part(similarity_tensor)  # (2 * batch_size, )
        result = reduce_mean(positive_tensor)

        return result

    
    def call(self, y_true, y_pred):
        '''
        calculate NT-Xent Loss
        input:
            1) y_true: 1st view tensor: (batch_size, # features * embebdding dim)
            2) y_pred: 2nd view tensor: (batch_size, # features * embebdding dim)
        output: NT-Xent loss
        '''
        # combine representation
        combined_representation = combine_representations(y_true, y_pred)
        
        # calculate cosine similarity
        similarity = calculate_similarity(combined_representation)

        # get positive pair index for each augmented sequence
        batch_size = y_true.shape[0]
        positive_indicies = generate_simclr_positive_indices(batch_size)

        # calculate NT-Xent loss
        similarity = divide_no_nan(similarity, tf.cast(self.temperature, dtype='float32'))
        similarity = set_diag(similarity, tf.fill(similarity.shape[0:-1], finfo('float32').min))
        # similarity = tf.cast(-log(softmax(similarity, axis=1)), dtype='float32') -> gradient returns nan value
        similarity = -log_softmax(similarity, axis=1)
        nt_xent_loss = self.calculate_NTXentLoss(similarity, positive_indicies)

        return tf.cast(nt_xent_loss, dtype='float32')
    
    
class AutoEncoderLoss(Loss):
    def __init__(self):
        '''Class for AutoEncoder Loss '''
        super(AutoEncoderLoss, self).__init__()
        self.mse_loss = MeanSquaredError(reduction='sum_over_batch_size')
   
    def call(self, y_true, y_pred):
        '''
        calculate NT-Xent Loss
        input:
            1) y_true: input tensor list for auto encoder [view 1, view 2]  
                -> [(batch_size, # of features * embed_dim), (batch_size, # of features * embed_dim)]
            2) y_pred: output tensor list for auto encoder [view 1, view 2]  
                -> [(batch_size, # of features * embed_dim), (batch_size, # of features * embed_dim)]
        output: Mean Squared Error
        '''
        first_view_mse_loss = self.mse_loss(y_true[0], y_pred[0])
        second_view_mse_loss = self.mse_loss(y_true[1], y_pred[1])
        return first_view_mse_loss + second_view_mse_loss          


class RootMeanSquaredLoss(Loss):
    '''Class for Root Mean Sqaured Loss '''
    def __init__(self):
        super(RootMeanSquaredLoss, self).__init__()
        self.mse_loss = MeanSquaredError(reduction='sum_over_batch_size')
   
    def call(self, y_true, y_pred):
        '''
        calculate Root Mean Squared Error(RMSE)
        input:
            1) y_true: real rating tensor (batch size, )  
            2) y_pred: predicted rating tensor (batch size, )
        output: Root Mean Squared Error
        '''
        mse_loss = self.mse_loss(y_true=y_true, y_pred=y_pred)
        return sqrt(mse_loss)  