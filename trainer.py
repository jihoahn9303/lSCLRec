from typing import List
import numpy as np
import tensorflow as tf 
from tensorflow import clip_by_norm
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss 
from tensorflow.keras.metrics import Metric, Mean
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import Sequence
import wandb
from loguru import logger


class ContrastiveLearningTrainer():
    def __init__(
        self,
        model: Model,
        nt_xent_loss: Loss,
        auto_encoder_loss: Loss,
        metric: Metric,
        lambda_for_cl: float,
        optimizer: Optimizer,
        dataloader: List[Sequence]   
    ):
        '''
        Class for trainer with using contrastive learning 
        input:
            1) model: model that will be trained
            2) nt_xent_loss: Instance for NT-Xent loss class
            3) auto_encoder_loss: Instance for AutoEncoder loss class
            4) metric: Metric for contrastive learning
            5) lambda_for_cl: Value to control loss for encoder model
            6) optimizer: Optimizer instance
            7) dataloader: List for tensors which contain train/valid and test dataloader
        
        '''
        self.model = model 
        self.nt_xent_loss = nt_xent_loss 
        self.auto_encoder_loss = auto_encoder_loss 
        self.metric = metric 
        self.mean_metric = Mean()
        self.lambda_for_cl = lambda_for_cl
        self.optimizer = optimizer
        self.dataloader = dataloader  # [train dataloader, valid dataloader, test dataloader]
        
    
    def train(
        self,
        clip_norm: float,
        log_interval: int,
        num_epochs: int,
        limit_patience: int,
        path: str
    ):
        '''
        train contrastive learning framework
        input:
            1) clip_norm: maximum norm for clipping gradient
            2) log_interval: interval for logging loss into wandb flatform
            3) num_epochs: number of epoch for training and validation
            4) limit_patience: early stopping patience
            5) path: path for saving best model
        '''        
        global_step = 0
        patience = 0
        best_validation_loss = np.float32(1e9)
        self.metric.reset_state()
        self.mean_metric.reset_state()
        # steps_per_epoch = len(self.dataloader[0])
            
        for epoch in range(num_epochs):
            ##### train step ##### 
            for step, item_indices in enumerate(self.dataloader[0]):
                with tf.GradientTape() as tape:
                    auto_encoder_input_1, auto_encoder_output_1 = self.model(item_indices[0], training=True)
                    auto_encoder_input_2, auto_encoder_output_2 = self.model(item_indices[1], training=True)
                    
                    # calculate NT-Xent loss
                    NT_XENT_LOSS = self.nt_xent_loss(auto_encoder_output_1, auto_encoder_output_2)

                    # calculate Auto encoder loss
                    AUTO_ENC_LOSS = self.auto_encoder_loss(
                        y_true=[auto_encoder_input_1, auto_encoder_input_2], 
                        y_pred=[auto_encoder_output_1, auto_encoder_output_2]
                    )

                    # calculate total loss
                    TOTAL_LOSS = NT_XENT_LOSS + tf.cast(self.lambda_for_cl, dtype='float32') * AUTO_ENC_LOSS

                
                # compute gradients
                trainable_vars = self.model.trainable_variables 
                gradients = tape.gradient(TOTAL_LOSS, trainable_vars)  
                
                # gradient clipping
                # gradients = [(tf.clip_by_norm(grad, clip_norm=clip_norm)) for grad in gradients]
                
                # update parameters 
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                
                # update metric
                self.metric.update_state(auto_encoder_output_1, auto_encoder_output_2)
                self.mean_metric.update_state(self.metric.result())
                self.metric.reset_state()
        
                lr = self.optimizer.learning_rate(global_step).numpy()  # get learning rate from optimizer 
                
                # logging with loguru
                logger.info(
                    f"Train: "
                    f"[{epoch}/{num_epochs}][{step}/{len(self.dataloader[0])}]\t"
                    f"lr {lr:.4f}\t"
                    f"total loss {TOTAL_LOSS.numpy():.6f}\t"
                    f"metric loss {self.mean_metric.result().numpy():.6f}"
                )
                            
                # synchronize train log for metric per log interval step(wandb)        
                if not (global_step + 1) % log_interval:
                    mean_metric_loss = self.mean_metric.result()
                    wandb.log(
                        data={
                            "train": {
                                "lr": lr, 
                                "metric_loss": mean_metric_loss.numpy(), 
                                "dense_1_gradient": wandb.Histogram(gradients[0].numpy()),
                                "dense_2_gradient": wandb.Histogram(gradients[4].numpy()),
                                "dense_3_gradient": wandb.Histogram(gradients[8].numpy()),
                                "dense_4_gradient": wandb.Histogram(gradients[12].numpy()),
                            },
                        },
                        step=global_step,
                        commit=False if step == len(self.dataloader[0]) - 1 else True,
                    )
                    
                    # reset metric
                    self.mean_metric.reset_state()
                                   
                global_step += 1              
        
            ##### validation step #####
            self.metric.reset_state()
            self.mean_metric.reset_state()
            
            for step, item_indices in enumerate(self.dataloader[1]):
                auto_encoder_input_1, auto_encoder_output_1 = self.model(item_indices[0], training=False)
                auto_encoder_input_2, auto_encoder_output_2 = self.model(item_indices[1], training=False)
                
                # calculate NT-Xent loss
                NT_XENT_LOSS = self.nt_xent_loss(auto_encoder_output_1, auto_encoder_output_2)

                # calculate Auto encoder loss
                AUTO_ENC_LOSS = self.auto_encoder_loss(
                    y_true=[auto_encoder_input_1, auto_encoder_input_2], 
                    y_pred=[auto_encoder_output_1, auto_encoder_output_2]
                )

                # calculate total loss
                TOTAL_LOSS = NT_XENT_LOSS + tf.cast(self.lambda_for_cl, dtype='float32') * AUTO_ENC_LOSS
                
                # update metric
                self.metric.update_state(auto_encoder_output_1, auto_encoder_output_2)
                self.mean_metric.update_state(self.metric.result())
                self.metric.reset_state()
                
                # logging with loguru
                logger.info(
                    f"Validation: "
                    f"[{epoch}/{num_epochs}][{step}/{len(self.dataloader[1])}]\t"
                    f"loss {TOTAL_LOSS.numpy():.6f}\t"
                    f"metric loss {self.mean_metric.result().numpy():.6f}"
                )
                
            # synchronize validation log to wandb
            wandb.log(
                data={"validation": {'metric_loss': self.mean_metric.result().numpy()}},
                step=global_step - 1,
                commit=True,
            )     
                
            
            # patience code
            mean_metric_loss = self.mean_metric.result().numpy()
            if np.float32(mean_metric_loss) < best_validation_loss:
                # update best metric loss
                best_validation_loss = np.float32(mean_metric_loss)
                
                # save model (in SavedModel format)
                self.model.save(f"{path}/best_model")
                
                # initialize patience value
                patience = 0
            else:
                patience += 1
            
            # reset metric value 
            self.metric.reset_state()
            self.mean_metric.reset_state()  
              
            if patience == limit_patience:
                break
            
                        
    def test(self, path, repeat_num: int = 10):
        load_model = tf.keras.models.load_model(f"{path}/best_model")
        
        total_metric_loss: float = 0.
        for i in range(repeat_num):
            self.metric.reset_state()
            self.mean_metric.reset_state()
        
            for item_indices in self.dataloader[2]:
                _, auto_encoder_output_1 = load_model(item_indices[0], training=False)
                _, auto_encoder_output_2 = load_model(item_indices[1], training=False)
                
                # update metric
                self.metric.update_state(auto_encoder_output_1, auto_encoder_output_2)
                self.mean_metric.update_state(self.metric.result())
                self.metric.reset_state()
                
            # logging with loguru
            logger.info(
                f"{i+1}-th Test: "
                f"metric loss {self.mean_metric.result().numpy():.6f}"
            )
            
            total_metric_loss += self.mean_metric.result().numpy()
        
        total_metric_loss /=  np.float32(repeat_num)
        
        # synchronize test log to wandb
        wandb.log(
            # data={"test": {'metric_loss': self.mean_metric.result().numpy()}},
            data={"test": {'metric_loss': total_metric_loss}},
            commit=True
        )
        
        self.metric.reset_state()
        self.mean_metric.reset_state()     
            

### recommdentation system trainer ###           
class RecommenderTrainer():
    def __init__(
        self,
        model: Model,
        rmse_loss: Loss,
        metric: Metric,
        optimizer: Optimizer,
        dataloader: List[Sequence]   
    ):
        self.model = model 
        self.rmse_loss = rmse_loss 
        self.metric = metric 
        self.mean_metric = Mean()
        self.optimizer = optimizer
        self.dataloader = dataloader  # [train dataloader, valid dataloader, test dataloader]
        '''
        Class for Recommender trainer
        input:
            1) model: model that will be trained
            2) rmse_loss: Instance for Root Mean Squared Error Loss class
            3) metric: Metric for Recommendation system
            4) optimizer: Optimizer instance
            5) dataloader: List for tensors which contain train/valid and test dataloader
        
        '''
    
    def train(
        self,
        clip_norm: float,
        log_interval: int,
        num_epochs: int,
        limit_patience: int,
        path: str
    ):
        '''
        train contrastive learning framework
        input:
            1) clip_norm: maximum norm for clipping gradient
            2) log_interval: interval for logging loss into wandb flatform
            3) num_epochs: number of epoch for training and validation
            4) limit_patience: early stopping patience
            5) path: path for saving best model
        '''        
        global_step = 0
        patience = 0
        best_validation_loss = np.float32(1e9)
        self.metric.reset_state()
        self.mean_metric.reset_state()
            
        for epoch in range(num_epochs):
            ##### train step ##### 
            for step, (user_item_index_set, ratings) in enumerate(self.dataloader[0]):
                user_indices = tf.reshape(user_item_index_set[0], shape=[user_item_index_set[0].shape[0], -1])
                item_indices = tf.reshape(user_item_index_set[1], shape=[user_item_index_set[1].shape[0], -1])
                ratings = tf.reshape(ratings, shape=[ratings.shape[0], -1])
                inputs = tf.concat([user_indices, item_indices, ratings], axis=-1)
                
                with tf.GradientTape() as tape:
                    # pass input to model
                    rating_tensor, projection_head_output = self.model(inputs, training=True)
                    
                    # calculate rmse loss
                    RMSE_LOSS = self.rmse_loss(rating_tensor, projection_head_output)
               
                # compute gradients
                trainable_vars = self.model.trainable_variables 
                gradients = tape.gradient(RMSE_LOSS, trainable_vars)  
                
                # update parameters 
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                
                # update metric
                self.metric.update_state(rating_tensor, projection_head_output)
                self.mean_metric.update_state(self.metric.result())
                self.metric.reset_state()
        
                lr = self.optimizer.learning_rate(global_step).numpy()  # get learning rate from optimizer 
                
                # logging with loguru
                logger.info(
                    f"Train: "
                    f"[{epoch}/{num_epochs}][{step}/{len(self.dataloader[0])}]\t"
                    f"lr {lr:.4f}\t"
                    f"train loss {RMSE_LOSS.numpy():.6f}\t"
                    f"metric loss {self.mean_metric.result().numpy():.6f}"
                )
                            
                # synchronize train log for metric per log interval step(wandb)        
                if not (global_step + 1) % log_interval:
                    mean_metric_loss = self.mean_metric.result()
                    wandb.log(
                        data={
                            "train": {
                                "lr": lr, 
                                "metric_loss": mean_metric_loss.numpy(), 
                                # "dense_1_gradient": wandb.Histogram(gradients[0].numpy()),
                            },
                        },
                        step=global_step,
                        commit=False if step == len(self.dataloader[0]) - 1 else True,
                    )
                    
                    # reset metric
                    self.mean_metric.reset_state()
                                   
                global_step += 1              
        
            ##### validation step #####
            self.metric.reset_state()
            self.mean_metric.reset_state()
            
            for step_valid, (user_item_index_set_valid, ratings_valid) in enumerate(self.dataloader[1]):
                user_indices_valid = tf.reshape(user_item_index_set_valid[0], shape=[user_item_index_set_valid[0].shape[0], -1])
                item_indices_valid = tf.reshape(user_item_index_set_valid[1], shape=[user_item_index_set_valid[1].shape[0], -1])
                ratings_valid = tf.reshape(ratings_valid, shape=[ratings_valid.shape[0], -1])
                inputs_valid = tf.concat([user_indices_valid, item_indices_valid, ratings_valid], axis=-1)
                
                # pass input to model
                rating_tensor_valid, projection_head_output_valid = self.model(inputs_valid, training=False)
                
                # calculate rmse
                RMSE_LOSS = self.rmse_loss(rating_tensor_valid, projection_head_output_valid)
                
                # update metric
                self.metric.update_state(rating_tensor_valid, projection_head_output_valid)
                self.mean_metric.update_state(self.metric.result())
                self.metric.reset_state()
                
                # logging with loguru
                logger.info(
                    f"Validation: "
                    f"[{epoch}/{num_epochs}][{step_valid}/{len(self.dataloader[1])}]\t"
                    f"valid loss {RMSE_LOSS.numpy():.6f}\t"
                    f"metric loss {self.mean_metric.result().numpy():.6f}"
                )
                
            # synchronize validation log to wandb
            wandb.log(
                data={"validation": {'metric_loss': self.mean_metric.result().numpy()}},
                step=global_step - 1,
                commit=True,
            )     
                       
            # patience code
            mean_metric_loss = self.mean_metric.result().numpy()
            if np.float32(mean_metric_loss) < best_validation_loss:
                # update best metric loss
                best_validation_loss = np.float32(mean_metric_loss)
                
                # save model (in SavedModel format)
                self.model.save(f"{path}/best_model")
                
                # initialize patience value
                patience = 0
            else:
                patience += 1
            
            # reset metric value 
            self.metric.reset_state()
            self.mean_metric.reset_state()  
              
            if patience == limit_patience:
                break  
            
    
    def test(self, path, repeat_num: int = 30):
        load_model = tf.keras.models.load_model(f"{path}/best_model")
        
        self.metric.reset_state()
        self.mean_metric.reset_state()
    
        for user_item_index_set_test, ratings_test in self.dataloader[2]:
            user_indices_test= tf.reshape(user_item_index_set_test[0], shape=[user_item_index_set_test[0].shape[0], -1])
            item_indices_test = tf.reshape(user_item_index_set_test[1], shape=[user_item_index_set_test[1].shape[0], -1])
            ratings_test = tf.reshape(ratings_test, shape=[ratings_test.shape[0], -1])
            inputs_test = tf.concat([user_indices_test, item_indices_test, ratings_test], axis=-1)
            
            # pass input to model
            rating_tensor_test, projection_head_output_test = load_model(inputs_test, training=False)
            
            # update metric
            self.metric.update_state(rating_tensor_test, projection_head_output_test)
            self.mean_metric.update_state(self.metric.result())
            self.metric.reset_state()
                
            # logging with loguru
            # logger.info(
            #     f"Test: "
            #     f"metric loss {self.mean_metric.result().numpy():.6f}"
            # )
            logger.info(
                f"Test: "
                f"metric loss(MAE) {self.mean_metric.result().numpy():.6f}"
            )
        
        # synchronize test log to wandb
        # wandb.log(
        #     data={"test": {'metric_loss': self.mean_metric.result().numpy()}},
        #     commit=True
        # )
        wandb.log(
            data={"test": {'MAE': self.mean_metric.result().numpy()}},
            commit=True
        )
        
        self.metric.reset_state()
        self.mean_metric.reset_state() 
            
    
     