import tensorflow as tf 
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout, ReLU 
from tensorflow.math import reduce_mean

AUTO_ENCODER_NAME = 'AutoEncoder'
PROJECTION_HEAD_NAME = 'MLP-Based-Projection-head'
MLP_NAME = 'MLP_NCF'

# make custom global average pooling layer with masking
class GlobalAveragePoolingWithMask(Layer):
    def call(self, inputs, mask=None):
        broadcast_float_mask = tf.expand_dims(tf.cast(mask, "float32"), axis=-1)  # (user_len, item_len, 1)
        broadcast_mul = broadcast_float_mask * inputs  # (user_len, item_len, embed_dim)
        result = reduce_mean(broadcast_mul, axis=1)  # (user_len, embed_dim)
        
        return result
    
    
class AutoEncoder(Layer):
    '''
    2-Layer stacked auto encoder
    Dense -> Batch Normalization -> ReLU -> (Dropout)
    Input -> hidden layer 1 -> hidden layer 2 -> hidden layer 1 -> Input
    '''
    def __init__(self, input_dim: int, dropout_rate: int):
        super(AutoEncoder, self).__init__(name=AUTO_ENCODER_NAME)
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
       
        self.dense_1 = Dense(units=self.input_dim//2, kernel_initializer='he_normal', name='Dense_1')
        self.batchnorm_1 = BatchNormalization(name='Batchnorm_1')
        self.dropout_1 = Dropout(rate=dropout_rate)
        self.activation_1 = ReLU()

        self.dense_2 = Dense(units=self.input_dim//4, kernel_initializer='he_normal', name='Dense_2')
        self.batchnorm_2 = BatchNormalization(name='Batchnorm_2')
        self.activation_2 = ReLU()

        self.dense_3 = Dense(units=self.input_dim//2, kernel_initializer='he_normal', name='Dense_3')
        self.batchnorm_3 = BatchNormalization(name='Batchnorm_3')
        self.activation_3 = ReLU()
        self.dropout_2 = Dropout(rate=dropout_rate)

        self.dense_4 = Dense(units=self.input_dim, kernel_initializer='he_normal', name='Dense_4')
        self.activation_4 = ReLU()


    def call(self, inputs, training):
        # inputs shape: (batch size, features * embedding dim)
        # output shape: (batch size, features * embedding dim)
        if training == False:
            self.dense_1.trainable = False
            self.dense_2.trainable = False
            self.dense_3.trainable = False
            self.dense_4.trainable = False
            self.batchnorm_1.trainable = False
            self.batchnorm_2.trainable = False
            self.batchnorm_3.trainable = False
            
                        
        else:  # training = True
            self.dense_1.trainable = True
            self.dense_2.trainable = True
            self.dense_3.trainable = True
            self.dense_4.trainable = True
            self.batchnorm_1.trainable = True
            self.batchnorm_2.trainable = True
            self.batchnorm_3.trainable = True
                          
        x = self.dense_1(inputs)
        x = self.batchnorm_1(x, training=training)
        x = self.activation_1(x)
        x = self.dropout_1(x, training=training)
        
        x = self.dense_2(x)
        x = self.batchnorm_2(x, training=training)
        x = self.activation_2(x)

        x = self.dense_3(x)
        x = self.batchnorm_3(x, training=training)
        x = self.activation_3(x)
        x = self.dropout_2(x, training=training)

        x = self.dense_4(x)
        x = self.activation_4(x)

        return x  
    
    
    
class ProjectionHead(Layer):
    '''
    MLP-Based projection head (2-layer)
    Dense -> Batch Normalization -> ReLU -> (Dropout)
    Input -> Input//2 -> (Input//4) -> (Input//8) -> 1
    '''
    def __init__(self, input_dim: int, dropout_rate: int):
        super(ProjectionHead, self).__init__(name=PROJECTION_HEAD_NAME)
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
       
        self.dense_1 = Dense(units=self.input_dim//2, kernel_initializer='he_normal', name='REC_Dense_1')
        self.batchnorm_1 = BatchNormalization(name='REC_Batchnorm_1')
        self.dropout_1 = Dropout(rate=self.dropout_rate)
        self.activation_1 = ReLU()

        self.dense_2 = Dense(units=self.input_dim//4, kernel_initializer='he_normal', name='REC_Dense_2')
        self.batchnorm_2 = BatchNormalization(name='REC_Batchnorm_2')
        self.activation_2 = ReLU()

        self.dense_3 = Dense(units=1, kernel_initializer='he_normal', name='REC_Dense_3')
        self.activation_3 = ReLU()



    def call(self, inputs, training):
        # inputs shape: (batch size, (user features + item features) * embedding dim)
        # output shape: (batch size, )
        if training == False:
            self.dense_1.trainable = False
            self.dense_2.trainable = False
            self.dense_3.trainable = False
            self.batchnorm_1.trainable = False
            self.batchnorm_2.trainable = False
                     
        else:  # training = True
            self.dense_1.trainable = True
            self.dense_2.trainable = True
            self.dense_3.trainable = True
            self.batchnorm_1.trainable = True
            self.batchnorm_2.trainable = True
                          
        x = self.dense_1(inputs)
        x = self.batchnorm_1(x, training=training)
        x = self.activation_1(x)
        x = self.dropout_1(x, training=training)
        
        x = self.dense_2(x)
        x = self.batchnorm_2(x, training=training)
        x = self.activation_2(x)

        x = self.dense_3(x)
        x = self.activation_3(x)

        return x
    
    
class MLP(Layer):
    '''
    MLP Layer
    Dense -> Batch Normalization -> ReLU -> (Dropout)
    Input -> Input//2 -> Input//4 -> Input//8 -> 1
    '''
    def __init__(self, input_dim: int, dropout_rate: int):
        super(MLP, self).__init__(name=MLP_NAME)
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
    
        self.dense_1 = Dense(units=self.input_dim//2, kernel_initializer='he_normal', name='NCF_Dense_1')
        self.batchnorm_1 = BatchNormalization(name='NCF_Batchnorm_1')
        self.dropout_1 = Dropout(rate=self.dropout_rate)
        self.activation_1 = ReLU()

        self.dense_2 = Dense(units=self.input_dim//4, kernel_initializer='he_normal', name='NCF_Dense_2')
        self.batchnorm_2 = BatchNormalization(name='NCF_Batchnorm_2')
        self.dropout_2 = Dropout(rate=self.dropout_rate)
        self.activation_2 = ReLU()
        
        # self.dense_3 = Dense(units=self.input_dim//8, kernel_initializer='he_normal', name='NCF_Dense_3')
        # self.batchnorm_3 = BatchNormalization(name='NCF_Batchnorm_3')
        # self.activation_3 = ReLU()

        self.dense_4 = Dense(units=1, kernel_initializer='he_normal', name='NCF_Dense_4')
        self.activation_4 = ReLU()
            
        
    def call(self, inputs, training):
        # inputs shape: (batch size, 2 * embedding dim)
        # output shape: (batch size, )
        if training == False:
            self.dense_1.trainable = False
            self.dense_2.trainable = False
            # self.dense_3.trainable = False
            self.dense_4.trainable = False
            self.batchnorm_1.trainable = False
            self.batchnorm_2.trainable = False
            # self.batchnorm_3.trainable = False
                     
        else:  # training = True
            self.dense_1.trainable = True
            self.dense_2.trainable = True
            # self.dense_3.trainable = True
            self.dense_4.trainable = True
            self.batchnorm_1.trainable = True
            self.batchnorm_2.trainable = True
            # self.batchnorm_3.trainable = True
                          
        x = self.dense_1(inputs)
        x = self.batchnorm_1(x, training=training)
        x = self.activation_1(x)
        x = self.dropout_1(x, training=training)
        
        x = self.dense_2(x)
        x = self.batchnorm_2(x, training=training)
        x = self.activation_2(x)
        x = self.dropout_2(x, training=training)

        # x = self.dense_3(x)
        # x = self.batchnorm_3(x, training=training)
        # x = self.activation_3(x)
        
        x = self.dense_4(x)
        x = self.activation_4(x)

        return x