import warnings 

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# load environmental variables in .env file(wandb secret key)
load_dotenv()

from pathlib import Path 
import os
import pandas as pd
import numpy as np
import pickle
from src.utils import (
    calculate_transaction_num,
    split_user_index,
    define_vip_user,
    get_long_tail_item_list,
    get_item_category,
    get_user_age,
    get_user_height, 
    get_user_weight,
    select_user_by_sparsity
)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import load_model
from src.model import ISimContrastiveLearning, Recommender, OnlyRecommender, NCF
from src.dataset import IsimCLRDataLoader, RECDataLoader
from src.loss import NTXentLoss, AutoEncoderLoss, RootMeanSquaredLoss 
from src.metric import MetricForCL
from src.scheduler import LinearWarmLRSchedule
from trainer import ContrastiveLearningTrainer, RecommenderTrainer
from diversity import CLDiversity
from src.schema import load_config
import wandb 
from omegaconf import OmegaConf
from loguru import logger


### load config ###
config = load_config("config.yaml")

assert config.train_type in ['CL', 'REC', 'ONLY_REC', 'NCF']
assert config.run_mode in ['train', 'test', 'diversity']

TRAIN_TYPE = config.train_type  # 'CL', 'REC', 'ONLY_REC', 'NCF'
RUN_MODE = config.run_mode   # 'train', 'test', 'diversity'
TRY_NUM = config.try_num
DATA_TYPE = config.data_type  # 'RentTheRunway', 'MovieLens'
AUG_MODE = config.aug_mode  # 'substitute', 'mask', 'crop'

if (TRAIN_TYPE == 'CL') or (TRAIN_TYPE == 'REC'):
    RUN_NAME = f"{TRAIN_TYPE}-{RUN_MODE}-{AUG_MODE}-{TRY_NUM}-{DATA_TYPE}"
    GROUP_NAME = f"{TRAIN_TYPE}-{AUG_MODE}-{DATA_TYPE}"
    
else:  # TRAIN_TYPE = 'ONLY_REC' or 'NCF'
    RUN_NAME = f"{TRAIN_TYPE}-{RUN_MODE}-{TRY_NUM}-{DATA_TYPE}"
    GROUP_NAME = f"{TRAIN_TYPE}-{DATA_TYPE}"


### initialize wandb server ###
wandb.init(
    project="KHU_graduate",
    group=GROUP_NAME,
    name=RUN_NAME,
    mode="online",
)

### update configuration to wandb ###
update_dict = OmegaConf.to_container(config)

if (TRAIN_TYPE == 'ONLY_REC') or (TRAIN_TYPE == 'NCF'):  
    pop_list = [
        'lambda_for_cl', 'substitution_rate', 'substitution_rate', 'masking_rate', 'cropping_rate',
        'temperature', 'aug_mode'
    ]
    for config_key in pop_list:
        update_dict.pop(config_key, None)
     
wandb.config.update(
    OmegaConf.to_container(config)
)

### make path for saving model(overall architecture) ###
if RUN_MODE == 'train':
    if (TRAIN_TYPE != 'ONLY_REC') and (TRAIN_TYPE != 'NCF'):   # 'CL', 'REC'
        Path(f"model/{DATA_TYPE}/{TRAIN_TYPE}/{AUG_MODE}/{TRY_NUM}").mkdir(
            parents=True, exist_ok=True
        )
    else:  # 'ONLY_REC', 'NCF'
        Path(f"model/{DATA_TYPE}/{TRAIN_TYPE}/{TRY_NUM}").mkdir(
            parents=True, exist_ok=True
        )

### save log into selected file (add) ###
if config.run_mode == 'train': 
    logger.add("logs/train/{time}.log")
elif config.run_mode == 'test':  # test mode
    logger.add("logs/test/{time}.log")
else:  # diversity mode
    logger.add("logs/diversity/{time}.log")   
logger.info(f"Configuration: {config}")


### define constant and load data ###
if DATA_TYPE == 'RentTheRunway':
    ### please modify path for your environment ###
    PATH = 'D:/KHU/graduate/data/RentTheRunway'
    TRANSACTION_PATH = os.path.join(PATH, 'fashion_transaction_data.csv')
    ITEM_CATEGORY_PATH = os.path.join(PATH, 'item_category_dict_renttherunway.pickle')
    MAXIMUM_SIMILARITY_PATH = os.path.join(PATH, 'sim_max_dict_renttherunway.pickle')  
    USER_ITEM_SEQUENCE_PATH = os.path.join(PATH, 'user_item_list_dict_renttherunway.pickle')
    NUM_ITEM_FEATURE = 2  # id, category 
    
    USER_AGE_PATH = os.path.join(PATH, 'user_age_dict_renttherunway.pickle')
    USER_HEIGHT_PATH = os.path.join(PATH, 'user_height_dict_renttherunway.pickle')
    USER_WEIGHT_PATH = os.path.join(PATH, 'user_weight_dict_renttherunway.pickle')
    NUM_USER_FEATURE = 4  # id, age, height, weight
    
    REC_TRAIN_PATH = os.path.join(PATH, 'REC_renttherunway_train.csv')
    REC_VALID_PATH = os.path.join(PATH, 'REC_renttherunway_valid.csv')
    REC_TEST_PATH = os.path.join(PATH, 'REC_renttherunway_test.csv')
      
    ### load file ###
    # 1. transaction data
    rent_the_runway_df = pd.read_csv(TRANSACTION_PATH).drop(columns=['Unnamed: 0'], axis=1)

    # 2. item list for user
    with open(USER_ITEM_SEQUENCE_PATH, 'rb') as fw:
        USER_ITEM_SEQUENCE_DICT = pickle.load(fw)
        
    # 3. item category dictionary
    with open(ITEM_CATEGORY_PATH, 'rb') as fw:
        ITEM_CATEGORY_DICT = pickle.load(fw)
        
    # 4. maximum similarity dictionary for item
    with open(MAXIMUM_SIMILARITY_PATH, 'rb') as fw:
        MAX_ITEM_SIMILARITY_DICT = pickle.load(fw)
        
    NUM_USER = rent_the_runway_df['user_id'].nunique()
    NUM_ITEM = rent_the_runway_df['item_id'].nunique()
    NUM_CATEGORY = rent_the_runway_df['category'].nunique()
    
    if TRAIN_TYPE != 'CL':     
        NUM_USER_AGE = rent_the_runway_df['age'].nunique()
        NUM_USER_HEIGHT = rent_the_runway_df['height_inches'].nunique()
        NUM_USER_WEIGHT = rent_the_runway_df['weight'].nunique()
        
        # 5. user age dictionary
        with open(USER_AGE_PATH, 'rb') as fw:
            USER_AGE_DICT = pickle.load(fw)
            
        # 6. user height dictionary
        with open(USER_HEIGHT_PATH, 'rb') as fw:
            USER_HEIGHT_DICT = pickle.load(fw)
            
        # 7. user weight dictionary
        with open(USER_WEIGHT_PATH, 'rb') as fw:
            USER_WEIGHT_DICT = pickle.load(fw)
            
        # 8. train/valid/test dataframe (user id, item id, rating)
        REC_TRAIN_DF = pd.read_csv(REC_TRAIN_PATH).drop(columns=['Unnamed: 0'], axis=1)
        REC_VALID_DF = pd.read_csv(REC_VALID_PATH).drop(columns=['Unnamed: 0'], axis=1)
        REC_TEST_DF = pd.read_csv(REC_TEST_PATH).drop(columns=['Unnamed: 0'], axis=1)
        
        if RUN_MODE == 'diversity':
            TEST_USER_INDICES = rent_the_runway_df['user_id'].unique()  # numpy array
            LONG_TAIL_ITEM_LIST = get_long_tail_item_list(rent_the_runway_df, k=0.7)
            WHOLE_ITEM_LIST = rent_the_runway_df['item_id'].unique()
               
        
### load data loader ###
if TRAIN_TYPE == 'CL':
    if DATA_TYPE == 'RentTheRunway':
        transaction_num_user = calculate_transaction_num(rent_the_runway_df, 'user')
    
    ## 'MovieLens'일 때도 조건문 추가
    # elif DATA_TYPE == 'MovieLens':
    #     pass
    
    # make transaction count dataframe   
    transaction_num_user['is_vip'] =\
        transaction_num_user['Transaction Count'].map(lambda x: define_vip_user(x, threshold=transaction_num_user['Transaction Count'].quantile(q=0.9)))
    
    # split dataframe 
    train_idx, valid_idx, test_idx = split_user_index(
        transaction_num_user,
        test_size=0.1,
        valid_size=1/8
    )
    
    # load train, valid, test data loader
    train_loader = IsimCLRDataLoader(
        user_indices=train_idx, 
        batch_size=config.batch_size, 
        max_item_similarity_dict=MAX_ITEM_SIMILARITY_DICT, 
        item_sequence_dict=USER_ITEM_SEQUENCE_DICT,
        aug_mode=config.aug_mode,
        data_mode='train',
        sequence_len=config.sequence_len,
        substitute_rate=config.substitution_rate,
        mask_rate=config.masking_rate,
        crop_rate=config.cropping_rate,
        shuffle=True
    )

    valid_loader = IsimCLRDataLoader(
        user_indices=valid_idx, 
        batch_size=config.batch_size, 
        max_item_similarity_dict=MAX_ITEM_SIMILARITY_DICT, 
        item_sequence_dict=USER_ITEM_SEQUENCE_DICT,
        aug_mode=config.aug_mode,
        data_mode='valid',
        sequence_len=config.sequence_len,
        substitute_rate=config.substitution_rate,
        mask_rate=config.masking_rate,
        crop_rate=config.cropping_rate,
        shuffle=False
    )
    
    test_loader = IsimCLRDataLoader(
        user_indices=test_idx, 
        batch_size=config.batch_size, 
        max_item_similarity_dict=MAX_ITEM_SIMILARITY_DICT, 
        item_sequence_dict=USER_ITEM_SEQUENCE_DICT,
        aug_mode=config.aug_mode,
        data_mode='test',
        sequence_len=config.sequence_len,
        substitute_rate=config.substitution_rate,
        mask_rate=config.masking_rate,
        crop_rate=config.cropping_rate,
        shuffle=False
    )
       
else:   # TRAIN_TYPE = 'REC' or 'ONLY_REC' or 'NCF'
    if RUN_MODE != 'diversity':
        # load train, valid, test data loader
        train_loader = RECDataLoader(REC_TRAIN_DF, batch_size=config.batch_size, shuffle=True)
        valid_loader = RECDataLoader(REC_VALID_DF, batch_size=config.batch_size, shuffle=False)
        test_loader = RECDataLoader(REC_TEST_DF, batch_size=config.batch_size, shuffle=False)
    
    
### initialize model ###
if DATA_TYPE == 'RentTheRunway':
    if TRAIN_TYPE == 'CL':  # contrastive learning
        cl_model = ISimContrastiveLearning(
            embedding_name=['ITEM_ID', 'ITEM_CATEGORY'],
            embedding_size=[NUM_ITEM, NUM_CATEGORY],
            embedding_dim=config.embedding_dim,
            num_item_feature=NUM_ITEM_FEATURE,
            dropout_rate=config.dropout_rate,
            dictionary_func_list=[get_item_category],
            feature_dict_list=[ITEM_CATEGORY_DICT], 
        )
    elif TRAIN_TYPE == 'REC':  # recommendation model learning
        if AUG_MODE == 'substitute':
            ### please modify path for your environment ###
            CL_MODEL_PATH = 'D:/KHU/graduate/model/RentTheRunway/CL/substitute/30/best_model'   
        elif AUG_MODE == 'mask':
            ### please modify path for your environment ###
            CL_MODEL_PATH = 'D:/KHU/graduate/model/RentTheRunway/CL/mask/9/best_model'          
        else:  # 'crop'
            ### please modify path for your environment ###
            CL_MODEL_PATH = 'D:/KHU/graduate/model/RentTheRunway/CL/crop/2/best_model'         
        rec_model = Recommender(
            cl_model_path=CL_MODEL_PATH,
            user_embedding_name=['USER_ID', 'USER_AGE', 'USER_HEIGHT', 'USER_WEIGHT'],
            user_feature_embedding_size=[NUM_USER, NUM_USER_AGE, NUM_USER_HEIGHT, NUM_USER_WEIGHT],
            embedding_dim=config.embedding_dim,
            num_item_feature=NUM_ITEM_FEATURE,
            num_user_feature=NUM_USER_FEATURE,
            rec_dropout_rate=config.dropout_rate,
            user_dictionary_func_list=[get_user_age, get_user_height, get_user_weight],
            user_feature_dict_list=[USER_AGE_DICT, USER_HEIGHT_DICT, USER_WEIGHT_DICT],
            item_sequence_len=config.sequence_len
        )
    elif TRAIN_TYPE == 'ONLY_REC': 
        rec_model = OnlyRecommender(
            user_embedding_name=['USER_ID', 'USER_AGE', 'USER_HEIGHT', 'USER_WEIGHT'],
            item_embedding_name=['ITEM_ID', 'ITEM_CATEGORY'],
            user_feature_embedding_size=[NUM_USER, NUM_USER_AGE, NUM_USER_HEIGHT, NUM_USER_WEIGHT],
            item_feature_embedding_size=[NUM_ITEM, NUM_CATEGORY],
            embedding_dim=config.embedding_dim,
            num_item_feature=NUM_ITEM_FEATURE,
            num_user_feature=NUM_USER_FEATURE,
            dropout_rate=config.dropout_rate,
            user_dictionary_func_list=[get_user_age, get_user_height, get_user_weight],
            item_dictionary_func_list=[get_item_category],
            user_feature_dict_list=[USER_AGE_DICT, USER_HEIGHT_DICT, USER_WEIGHT_DICT],
            item_feature_dict_list=[ITEM_CATEGORY_DICT]
        )
    else:   # TRAIN_TYPE = 'NCF'
        rec_model = NCF(
            embedding_name=['USER_ID', 'ITEM_ID'],
            embedding_size=[NUM_USER, NUM_ITEM],
            embedding_dim=config.embedding_dim,
            dropout_rate=config.dropout_rate
        )


### load loss, metric, optimizer and trainer ###
if TRAIN_TYPE == 'CL':
    nt_xent_loss = NTXentLoss(temperature=config.temperature)  # NT-Xent Loss   
    auto_encoder_loss = AutoEncoderLoss() # Auto encoder Loss
    cl_metrics = MetricForCL(temperature=config.temperature)   # Metric for Contrastive Learning
    scheduler = LinearWarmLRSchedule(config.learning_rate, config.warmup_steps) # scheduler
    optimizer = Adam(learning_rate=scheduler)  # optimizer

    trainer = ContrastiveLearningTrainer(
        model=cl_model,
        nt_xent_loss=nt_xent_loss,
        auto_encoder_loss=auto_encoder_loss,
        metric=cl_metrics,
        lambda_for_cl=config.lambda_for_cl,
        optimizer=optimizer,
        dataloader=[train_loader, valid_loader, test_loader]
    )
    
else:  # TRAIN_TYPE = 'REC' or 'ONLY_REC' or 'NCF'
    if RUN_MODE != 'diversity': 
        rec_rmse_loss = RootMeanSquaredLoss()
        # rec_metrics = RootMeanSquaredError()   # Metric for Recommendation model learning
        rec_metrics = MeanAbsoluteError()
        scheduler = LinearWarmLRSchedule(config.learning_rate, config.warmup_steps) # scheduler
        optimizer = Adam(learning_rate=scheduler)  # optimizer
        
        trainer = RecommenderTrainer(
            model=rec_model,
            rmse_loss=rec_rmse_loss,
            metric=rec_metrics,
            optimizer=optimizer,
            dataloader=[train_loader, valid_loader, test_loader]
        )
    
### handling model ###
if (TRAIN_TYPE != 'ONLY_REC') and (TRAIN_TYPE != 'NCF'):
    MODEL_SAVE_PATH = f"model/{DATA_TYPE}/{TRAIN_TYPE}/{AUG_MODE}/{TRY_NUM}"
else:
    MODEL_SAVE_PATH = f"model/{DATA_TYPE}/{TRAIN_TYPE}/{TRY_NUM}" 
        
if RUN_MODE == 'train': 
    trainer.train(
        clip_norm=config.clip_norm,
        log_interval=config.log_interval, 
        num_epochs=config.epochs, 
        limit_patience=config.early_stop_patience,
        path=MODEL_SAVE_PATH
    )

elif RUN_MODE == 'test': 
    trainer.test(
        path=MODEL_SAVE_PATH,
        repeat_num=30
    )
    
else:  # RUN_MODE = 'diversity'
    # 1. load model
    if TRAIN_TYPE == 'REC':
        if AUG_MODE == 'substitute':
            ### please modify path for your environment ###
            REC_MODEL_PATH = 'D:/KHU/graduate/model/RentTheRunway/REC/substitute/3/best_model'  
        elif AUG_MODE == 'mask':
            ### please modify path for your environment ###
            REC_MODEL_PATH = 'D:/KHU/graduate/model/RentTheRunway/REC/mask/5/best_model'        
        else:  # crop
            ### please modify path for your environment ###
            REC_MODEL_PATH = 'D:/KHU/graduate/model/RentTheRunway/REC/crop/5/best_model'        
    
    elif TRAIN_TYPE == 'ONLY_REC':    
        ### please modify path for your environment ###
        REC_MODEL_PATH = 'D:/KHU/graduate/model/RentTheRunway/ONLY_REC/2/best_model'
        
    elif TRAIN_TYPE == 'NCF':
        ### please modify path for your environment ###
        REC_MODEL_PATH = 'D:/KHU/graduate/model/RentTheRunway/NCF/9/best_model'
    
    trained_model = load_model(REC_MODEL_PATH)        
    
    # 2. define test user
    FINAL_TEST_USER_INDICES = select_user_by_sparsity(
        df=rent_the_runway_df,
        test_user_index=TEST_USER_INDICES,
        sparsity=config.sparsity
    )
    
    # 3. load diversity class
    diversity = CLDiversity(
        trained_model=trained_model,
        user_indices=FINAL_TEST_USER_INDICES,
        whole_test_items=WHOLE_ITEM_LIST,
        long_tail_items=LONG_TAIL_ITEM_LIST,
        user_item_sequence_dict=USER_ITEM_SEQUENCE_DICT
    )
    
    # 4. calculate diversity
    long_tail_prediction_coverage = diversity.get_lpc(k=config.k)
    average_percentage_of_long_tail_items = diversity.get_aplt(k=config.k)
    
    # 5. synchronize log to wandb
    wandb.log(
        data={
            "diversity": {
                "LPC": long_tail_prediction_coverage, 
                "APLT": average_percentage_of_long_tail_items,
                "K": config.k
            },
        },
        commit=True
    )
    

    





