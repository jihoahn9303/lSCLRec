import tensorflow as tf 
from tensorflow.keras.optimizers.schedules import LearningRateSchedule 

class LinearWarmLRSchedule(LearningRateSchedule):
    def __init__(self, 
                 lr_peak: float,
                 warmup_end_steps: int
    ):
        super().__init__()
        self.lr_peak = lr_peak 
        self.warmup_end_steps = warmup_end_steps

    
    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        warmup_and_steps = tf.cast(self.warmup_end_steps, tf.float32) 
        lr_peak = tf.cast(self.lr_peak, tf.float32)

        return tf.cond(
            step_float < warmup_and_steps, # condition
            lambda: lr_peak * (tf.math.maximum(step_float, 1) / warmup_and_steps), # if true
            lambda: lr_peak  # else
        )