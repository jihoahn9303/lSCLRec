from typing import List 
import tensorflow as tf 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Concatenate
from .layer import AutoEncoder, GlobalAveragePoolingWithMask, ProjectionHead, MLP
from .utils import (
    make_padding_mask_tensor,
    get_embbeding_layer,
    make_lookup_table
)

CL_MODEL_NAME = 'ISimCLR'
REC_MODEL_NAME = 'REC_with_CLR'
ONLY_REC_MODEL_NAME = 'REC_without_CLR'
NCF_NAME = 'NCF_with_MLP'

class ISimContrastiveLearning(Model):
    def __init__(
        self,
        embedding_name: List[str],  # name for each embedding layer 
        embedding_size: List[int],  # unique count for each item feature
        embedding_dim: int,         # embedding size
        num_item_feature: int,      # number of feature used in item
        dropout_rate: int,
        dictionary_func_list,       # function list for item
        feature_dict_list: List[dict]
    ):
        '''
        feature_dict_list: feature의 이름에 대응하는 dictionary 객체
        dictionary_func_list: 특정 dictionary에서 item id에 대응하는 feature 값을 가져오는 함수
        CAUTION: 
            1) embedding_name(ID는 제외), dictionary_func_list와 feature_dict_list의 순서는 일치해야 함!
            2) embedding_name과 embedding_size의 순서는 일치해야 함!
        '''
        super(ISimContrastiveLearning, self).__init__(name=CL_MODEL_NAME)
        self.embedding_name = embedding_name  # ex: ['ITEM_CATEGORY', 'ITEM_PRICE']
        self.embedding_dim = embedding_dim
        self.embedding_size = embedding_size
        self.num_item_feature = num_item_feature
        self.dropout_rate = dropout_rate
        self.feature_dict_list = feature_dict_list  # ex: [ITEM_CATEGORY_DICT, ITEM_PRICE_DICT]
        self.dictionary_func_list = dictionary_func_list # ex: [get_item_category, get_item_price]
        self.autoencoder = AutoEncoder(
            input_dim=self.num_item_feature * self.embedding_dim,
            dropout_rate=self.dropout_rate, 
        )
        # get each embedding layer for coressponding to self.embedding_name
        self.embedding_layer_list = get_embbeding_layer( 
            embedding_name=self.embedding_name,
            embedding_dim=self.embedding_dim,
            embedding_size=self.embedding_size,
            # sequence_length=self.sequence_length,
        )
        # get each lookup table for coressponding to self.feature_list
        self.item_lookup_table_list = [
            make_lookup_table(info_dict=feature_dict, dictionary_func=self.dictionary_func_list[idx])
            for idx, feature_dict in enumerate(self.feature_dict_list)
            if len(self.feature_dict_list) != 0
        ]

    
    def call(
        self, 
        input, 
        training: bool = False
    ):
        '''
        input: 2-D tensor (batch size, length of item sequence)
        '''
        input_tensor = tf.cast(input, dtype='int32')
        
        # 1. get masking tensor
        mask_tensor = make_padding_mask_tensor(input_tensor)  # (batch size, length of item sequence)

        # 2. get lookuped tensor
        lookup_tensor_list = [input_tensor]

        if len(self.item_lookup_table_list) != 0:
            for lookup_table in self.item_lookup_table_list:
                lookup_tensor_list.append(
                    lookup_table.lookup(input_tensor)
                )

        # 3. pass lookup tensor to embedding layer -> (batch size, length of item sequence, embedding dim)
        if training == False:
            for embedding_layer in self.embedding_layer_list:
                embedding_layer.trainable = False
        else:
            for embedding_layer in self.embedding_layer_list:
                embedding_layer.trainable = True
        
        embbeded_tensor_list = [
            embedding_layer(lookup_tensor_list[layer_idx])
            for layer_idx, embedding_layer in enumerate(self.embedding_layer_list)
        ]
        
        
        # 4. global average pooling with masking tensor
        embbeded_list = [
            GlobalAveragePoolingWithMask()(embbeded_tensor, mask_tensor)
            for embbeded_tensor in embbeded_tensor_list
        ]

        # 5. concatenate tensors -> (batch size, embedding dim * number of features)
        concatenated_tensor = Concatenate(axis=-1)(embbeded_list) 

        # 6. pass concatenated tensor into auto encoder 
        autoencoder_output = self.autoencoder(concatenated_tensor, training=training)

        return concatenated_tensor, autoencoder_output  # return concatenated_tensor for calculate auto encoder loss
     
    
class Recommender(Model):
    def __init__(
        self,
        cl_model_path,   # path for trained contrastive learning model
        user_embedding_name: List[str],   # name for each embedding layer
        user_feature_embedding_size: List[int],  # unique count for user feature
        embedding_dim: int,     
        num_item_feature: int,   # number of item feature
        num_user_feature: int,   # number of user feature
        rec_dropout_rate: int,   # dropout rate for recommeder
        user_dictionary_func_list,  
        user_feature_dict_list: List[dict],  
        item_sequence_len: int   # sequence length for item
    ):
        '''
        user_feature_dict_list: user feature의 이름에 대응하는 dictionary 객체
        user_dictionary_func_list: 특정 dictionary에서 user id에 대응하는 feature 값을 가져오는 함수 리스트
        CAUTION: 
            1) user_embedding_name(ID는 제외), user_dictionary_func_list와 user_feature_dict_list의 순서는 일치해야 함!
            2) user_embedding_name과 user_feature_embedding_size의 순서는 일치해야 함!
        '''
        super(Recommender, self).__init__(name=REC_MODEL_NAME)
        self.user_embedding_name = user_embedding_name  # ex: ['USER_ID', 'USER_AGE', 'USER_HEIGHT', 'USER_WEIGHT']
        self.user_feature_embedding_size = user_feature_embedding_size
        self.embedding_dim = embedding_dim
        self.num_item_feature = num_item_feature
        self.num_user_feature = num_user_feature
        self.rec_dropout_rate = rec_dropout_rate
        self.user_dictionary_func_list = user_dictionary_func_list # ex: [get_user_age, get_user_height, get_user_weight]
        self.user_feature_dict_list = user_feature_dict_list  # ex: [USER_AGE_DICT, USER_HEIGHT_DICT, USER_WEIGHT_DICT]
        self.item_sequence_len = item_sequence_len
        self.projection_head = ProjectionHead(
            input_dim=(self.num_item_feature + self.num_user_feature) * self.embedding_dim,
            dropout_rate=self.rec_dropout_rate
        )
        
        # load pre-trained contrastive learning model
        self.cl_model = load_model(cl_model_path)  
        
        # get each embedding layer for coressponding to self.embedding_name
        self.user_embedding_layer_list = get_embbeding_layer( 
            embedding_name=self.user_embedding_name,
            embedding_dim=self.embedding_dim,
            embedding_size=self.user_feature_embedding_size
        )
        # get each lookup table for coressponding to self.feature_list
        self.user_lookup_table_list = [
            make_lookup_table(info_dict=user_feature_dict, dictionary_func=self.user_dictionary_func_list[idx])
            for idx, user_feature_dict in enumerate(self.user_feature_dict_list)
            if len(self.user_feature_dict_list) != 0
        ]

    
    def call(
        self, 
        input, 
        training: bool = False
    ):
        '''
        input: 2-D tensor (batch size, 3) <- (batch size, user id + item id + rating)
        '''
        user_id_tensor = tf.cast(input[:, 0], dtype='int32')
        user_id_tensor = tf.expand_dims(user_id_tensor, axis=-1)  # (batch size, 1)
        item_id_tensor = tf.cast(input[:, 1], dtype='int64')
        item_id_tensor = tf.expand_dims(item_id_tensor, axis=-1)  # (batch_size, 1)
        item_id_tensor = tf.repeat(input=item_id_tensor, repeats=[self.item_sequence_len], axis=1)  # (batch_size, length of item sequence)
        rating_tensor = tf.cast(input[:, 2], dtype='float32')
        
        ### handling pre-trained contrastive learning model ###
        self.cl_model.trainable = False   # freeze model
        _, cl_autoencoder_output = self.cl_model(item_id_tensor, training=False)  # cl_autoencoder_output: (batch_size, embedding dim * number of item feature)
        
        ### handling recommender model ###
        # 1. get lookuped tensor for user feature
        user_lookup_tensor_list = [user_id_tensor]

        if len(self.user_lookup_table_list) != 0:
            for user_lookup_table in self.user_lookup_table_list:
                user_lookup_tensor_list.append(
                    user_lookup_table.lookup(user_id_tensor)
                )

        # 2. pass lookuped tensor for user feature to embedding layer -> [(batch size, 1, embedding dim)] * number of user feature
        if training == False:
            for embedding_layer in self.user_embedding_layer_list:
                embedding_layer.trainable = False
        else:
            for embedding_layer in self.user_embedding_layer_list:
                embedding_layer.trainable = True
        
        embbeded_tensor_list = [
            embedding_layer(user_lookup_tensor_list[layer_idx])
            for layer_idx, embedding_layer in enumerate(self.user_embedding_layer_list)
        ]
        
        # 3. concatenate tensors 
        user_concatenated_tensor = Concatenate(axis=-1)(embbeded_tensor_list)  # (batch size, 1, embedding dim * number of user features)
        user_concatenated_tensor = tf.squeeze(user_concatenated_tensor, axis=1)  # (batch size, embedding dim * number of user features)

        # 4. concatenate user tensor and item tensor -> (batch size, embedding dim * (number of item feature + number of user features))
        final_concatenated_tensor = Concatenate(axis=-1)([cl_autoencoder_output, user_concatenated_tensor])
        
        ### pass concatenated tensor to projection head ###
        projection_head_output = self.projection_head(final_concatenated_tensor, training=training)

        return rating_tensor, projection_head_output  # return y_true, y_pred
    
    
       
class OnlyRecommender(Model):
    def __init__(
        self,
        user_embedding_name: List[str],
        item_embedding_name: List[str],
        user_feature_embedding_size: List[int],
        item_feature_embedding_size: List[int],
        embedding_dim: int,
        num_item_feature: int,
        num_user_feature: int,
        dropout_rate: int,
        user_dictionary_func_list,
        item_dictionary_func_list,
        user_feature_dict_list: List[dict],  
        item_feature_dict_list: List[dict],
    ):
        '''
        class for recommender system without contrastive learning framework
        '''
        super(OnlyRecommender, self).__init__(name=ONLY_REC_MODEL_NAME)
        self.user_embedding_name = user_embedding_name  # ex: ['USER_ID', 'USER_AGE', 'USER_HEIGHT', 'USER_WEIGHT']
        self.item_embedding_name = item_embedding_name  # ex: ['ITEM_ID', 'ITEM_CATEGORY']
        self.user_feature_embedding_size = user_feature_embedding_size
        self.item_feature_embedding_size = item_feature_embedding_size
        self.embedding_dim = embedding_dim
        self.num_item_feature = num_item_feature
        self.num_user_feature = num_user_feature
        self.dropout_rate = dropout_rate
        self.user_dictionary_func_list = user_dictionary_func_list # ex: [get_user_age, get_user_height, get_user_weight]
        self.item_dictionary_func_list = item_dictionary_func_list # ex: [get_item_category]
        self.user_feature_dict_list = user_feature_dict_list  # ex: [USER_AGE_DICT, USER_HEIGHT_DICT, USER_WEIGHT_DICT]
        self.item_feature_dict_list = item_feature_dict_list  # ex: [ITEM_CATEGORY_DICT]

        # deep autoencoder layer
        self.autoencoder = AutoEncoder(
            input_dim=self.num_item_feature * self.embedding_dim,
            dropout_rate=self.dropout_rate,
        )
        
        # MLP-based projection head layer
        self.projection_head = ProjectionHead(
            input_dim=(self.num_item_feature + self.num_user_feature) * self.embedding_dim,
            dropout_rate=self.dropout_rate
        )
              
        # get each embedding layer for coressponding to self.embedding_name
        self.user_embedding_layer_list = get_embbeding_layer( 
            embedding_name=self.user_embedding_name,
            embedding_dim=self.embedding_dim,
            embedding_size=self.user_feature_embedding_size
        )
        self.item_embedding_layer_list = get_embbeding_layer(
            embedding_name=self.item_embedding_name,
            embedding_dim=self.embedding_dim,
            embedding_size=self.item_feature_embedding_size
        )
        
        # get lookup table for coressponding to feature
        self.user_lookup_table_list = [
            make_lookup_table(info_dict=user_feature_dict, dictionary_func=self.user_dictionary_func_list[idx])
            for idx, user_feature_dict in enumerate(self.user_feature_dict_list)
            if len(self.user_feature_dict_list) != 0
        ]
        self.item_lookup_table_list = [
            make_lookup_table(info_dict=item_feature_dict, dictionary_func=self.item_dictionary_func_list[idx])
            for idx, item_feature_dict in enumerate(self.item_feature_dict_list)
            if len(self.item_feature_dict_list) != 0
        ]

    
    def call(
        self, 
        input, 
        training: bool = False
    ):
        '''
        input: 2-D tensor (batch size, 3) <- (batch size, user id + item id + rating)
        '''
        user_id_tensor = tf.cast(input[:, 0], dtype='int32')
        user_id_tensor = tf.expand_dims(user_id_tensor, axis=-1)  # (batch size, 1)
        item_id_tensor = tf.cast(input[:, 1], dtype='int32')
        item_id_tensor = tf.expand_dims(item_id_tensor, axis=-1)  # (batch_size, 1)
        rating_tensor = tf.cast(input[:, 2], dtype='float32')
        
        ### handling item features ###
        # 1. get lookuped tensor
        item_lookup_tensor_list = [item_id_tensor]

        if len(self.item_lookup_table_list) != 0:
            for item_lookup_table in self.item_lookup_table_list:
                item_lookup_tensor_list.append(
                    item_lookup_table.lookup(item_id_tensor)
                )

        # 2. pass lookup tensor to item embedding layer -> (batch size, 1, embedding dim) * number of item features
        if training == False:
            for item_embedding_layer in self.item_embedding_layer_list:
                item_embedding_layer.trainable = False
        else:
            for item_embedding_layer in self.item_embedding_layer_list:
                item_embedding_layer.trainable = True
        
        item_embbeded_tensor_list = [
            item_embedding_layer(item_lookup_tensor_list[layer_idx])
            for layer_idx, item_embedding_layer in enumerate(self.item_embedding_layer_list)
        ]
        
        # 3. concatenate tensors 
        item_concatenated_tensor = Concatenate(axis=-1)(item_embbeded_tensor_list)  # (batch size, 1, embedding dim * number of item features)
        item_concatenated_tensor = tf.squeeze(item_concatenated_tensor, axis=1)  # (batch size, embedding dim * number of item features)
             
        ### handling user features ###
        # 1. get lookuped tensor
        user_lookup_tensor_list = [user_id_tensor]

        if len(self.user_lookup_table_list) != 0:
            for user_lookup_table in self.user_lookup_table_list:
                user_lookup_tensor_list.append(
                    user_lookup_table.lookup(user_id_tensor)
                )

        # 2. pass lookup tensor to user embedding layer -> (batch size, 1, embedding dim)
        if training == False:
            for user_embedding_layer in self.user_embedding_layer_list:
                user_embedding_layer.trainable = False
        else:
            for user_embedding_layer in self.user_embedding_layer_list:
                user_embedding_layer.trainable = True
        
        user_embbeded_tensor_list = [
            user_embedding_layer(user_lookup_tensor_list[layer_idx])
            for layer_idx, user_embedding_layer in enumerate(self.user_embedding_layer_list)
        ]
        
        # 3. concatenate tensors 
        user_concatenated_tensor = Concatenate(axis=-1)(user_embbeded_tensor_list)  # (batch size, 1, embedding dim * number of user features)
        user_concatenated_tensor = tf.squeeze(user_concatenated_tensor, axis=1)  # (batch size, embedding dim * number of user features)
        
        
        ### handling concatenated item tensors ###
        autoencoder_output = self.autoencoder(item_concatenated_tensor, training=training)  # (batch size, embedding dim * number of item features)
        
        ### handling concatenated user tensors ###
        # (batch size, embedding dim * (number of item feature + number of user features))
        final_concatenated_tensor = Concatenate(axis=-1)([autoencoder_output, user_concatenated_tensor])
            
        ### handling projection head layer ###
        projection_head_output = self.projection_head(final_concatenated_tensor, training=training)  # (batch size, )

        return rating_tensor, projection_head_output  # return y_true, y_pred
    
    

class NCF(Model):
    def __init__(
        self,
        embedding_name: List[str],
        embedding_size: List[int],
        embedding_dim: int,
        dropout_rate: int,
    ):
        '''
        class for neural collaborative filtering
        '''
        super(NCF, self).__init__(name=NCF_NAME)
        self.embedding_name = embedding_name  # ['USER_ID', 'ITEM_ID']
        self.embedding_dim = embedding_dim
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        # get each embedding layer for coressponding to self.embedding_name
        self.embedding_layer_list = get_embbeding_layer( 
            embedding_name=self.embedding_name,
            embedding_dim=self.embedding_dim,
            embedding_size=self.embedding_size,
        )
        
        self.mlp_layer = MLP(
            input_dim=self.embedding_dim * 2,
            dropout_rate=self.dropout_rate
        )

 
    def call(
        self, 
        input, 
        training: bool = False
    ):
        '''
        input: 2-D tensor (batch size, 3) <- (batch size, user id + item id + rating)
        '''
        user_id_tensor = tf.cast(input[:, 0], dtype='int32')
        user_id_tensor = tf.expand_dims(user_id_tensor, axis=-1)  # (batch size, 1)
        item_id_tensor = tf.cast(input[:, 1], dtype='int32')
        item_id_tensor = tf.expand_dims(item_id_tensor, axis=-1)  # (batch_size, 1)
        rating_tensor = tf.cast(input[:, 2], dtype='float32')
        
        # 1. get lookup tensor
        lookup_tensor_list = [user_id_tensor, item_id_tensor]

        # 2. pass lookup tensor to embedding layer -> (batch size, 1, embedding dim) * 2
        if training == False:
            for embedding_layer in self.embedding_layer_list:
                embedding_layer.trainable = False
        else:
            for embedding_layer in self.embedding_layer_list:
                embedding_layer.trainable = True
        
        embbeded_tensor_list = [
            embedding_layer(lookup_tensor_list[layer_idx])
            for layer_idx, embedding_layer in enumerate(self.embedding_layer_list)
        ]
        
        # 3. concatenate tensors 
        concatenated_tensor = Concatenate(axis=-1)(embbeded_tensor_list)  # (batch size, 1, embedding dim * 2)
        concatenated_tensor = tf.squeeze(concatenated_tensor, axis=1)  # (batch size, embedding dim * 2)
                
        # 4. pass tensor to MLP Layer
        mlp_output = self.mlp_layer(concatenated_tensor, training=training)  # (batch size, )

        return rating_tensor, mlp_output  # return y_true, y_pred