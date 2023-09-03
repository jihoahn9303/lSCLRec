import tensorflow as tf 
from tensorflow.nn import softmax
from tensorflow.linalg import set_diag
from tensorflow.math import divide_no_nan
from tensorflow.experimental.numpy import finfo
from tensorflow.keras.metrics import Metric, SparseCategoricalCrossentropy
from .utils import (
    combine_representations,
    calculate_similarity,
    generate_simclr_positive_indices
)

class MetricForCL(tf.keras.metrics.Metric):
    
  def __init__(self, temperature : float, name='metric_for_contrastive_learning', **kwargs):
      super(MetricForCL, self).__init__(name=name, **kwargs)
      self.metric = self.add_weight(
          name='sparse_categorical_cross_entropy', 
          initializer='zeros'
      )
      self.sparse_categorical_cross_entropy = SparseCategoricalCrossentropy()
      self.temperature = temperature


  def update_state(self, y_true, y_pred, sample_weight=None):
      '''
      calculate sparse categorical cross entropy
      input:
          1) y_true: 1st view tensor: (batch_size, # features * embebdding dim)
          2) y_pred: 2nd view tensor: (batch_size, # features * embebdding dim)
      '''
      # combine representation
      combined_representation = combine_representations(y_true, y_pred)
      
      # calculate cosine similarity
      similarity = calculate_similarity(combined_representation)

      # get positive pair index for each augmented sequence
      batch_size = y_true.shape[0]
      positive_indicies = generate_simclr_positive_indices(batch_size)

      # calculate sparse categorical cross entropy
      similarity = divide_no_nan(similarity, tf.cast(self.temperature, dtype='float32'))
      similarity = set_diag(similarity, tf.fill(similarity.shape[0:-1], finfo('float32').min))
      similarity = softmax(similarity, axis=1)
      loss_value = self.sparse_categorical_cross_entropy(positive_indicies, similarity)

      # update loss value
      self.metric.assign_add(loss_value)


  def reset_state(self):
      self.metric.assign(0.)


  def result(self):
      return self.metric