# lSCLRec(Item Substitution Contrastive Learning Recommendation)

**대조 학습 및 아이템 대체를 활용하여 추천 시스템의 다양성을 높이는 졸업논문 주제입니다.**

본 주제에 대한 자세한 내용은 `대조 학습을 이용한 개인화 추천의 다양성 향상 연구.pdf` 파일에서 확인하실 수 있습니다!

## Develop & Experiment environment

개발 및 실험 환경에 대한 주요 사항은 아래와 같습니다.

|Source|Version|
|------|-------|
|OS| Microsoft Windows 10 Pro build `19045` | 
|Python| 3.9.13 | 
|Deep learning Framework| Tensorflow `2.10.0`|
|GPU| NVIDIA GeForce RTX 3060 12GB|
|Logging Interface| loguru `0.6.0`, Weight & Biases `0.13.10`|
|IDLE| Visual Studio code `1.75.1`|

## Project Architecture

본 프로젝트의 폴더 및 파일 구조는 다음과 같습니다.

```
 ┣ 📂.vscode
 ┃ ┗ 📜tasks.json
 ┣ 📂data
 ┃ ┗ 📂RentTheRunway
 ┃ ┃ ┣ 📜fashion_transaction_data.csv
 ┃ ┃ ┣ 📜item_category_dict_renttherunway.pickle
 ┃ ┃ ┣ 📜REC_renttherunway_test.csv
 ┃ ┃ ┣ 📜REC_renttherunway_train.csv
 ┃ ┃ ┣ 📜REC_renttherunway_valid.csv
 ┃ ┃ ┣ 📜sim_max_dict_renttherunway.pickle
 ┃ ┃ ┣ 📜user_age_dict_renttherunway.pickle
 ┃ ┃ ┣ 📜user_height_dict_renttherunway.pickle
 ┃ ┃ ┣ 📜user_item_list_dict_renttherunway.pickle
 ┃ ┃ ┗ 📜user_weight_dict_renttherunway.pickle
 ┣ 📂log
 ┣ 📂model
 ┣ 📂src
 ┃ ┣ 📜augmentation.py
 ┃ ┣ 📜dataset.py
 ┃ ┣ 📜layer.py
 ┃ ┣ 📜loss.py
 ┃ ┣ 📜metric.py
 ┃ ┣ 📜model.py
 ┃ ┣ 📜scheduler.py
 ┃ ┣ 📜schema.py
 ┃ ┣ 📜utils.py
 ┃ ┗ 📜__init__.py
 ┣ 📂wandb
 ┣ 📜.env.example
 ┣ 📜.gitignore
 ┣ 📜comfig.yaml
 ┣ 📜diversity.py
 ┣ 📜main.py
 ┣ 📜requirements.txt
 ┣ 📜test.py
 ┗ 📜trainer.py
```


## Usage

### Envirionment 

Anaconda prompt를 통해 본 프로젝트 폴더를 다운받은 위치에서 코드를 실행시키실 경우 아래와 같이 할 수 있습니다.

```text
python main.py
```

### IDLE

만약 Visual Studio Code, Pycharm과 같은 파이썬 통합 개발환경에서 실험에 필요한 인자를 직접 바꾸면서 수행하실 경우 `config.yaml` 파일에서 값을 변경한 후, `main.py`파일을 실행하시면 됩니다.

`config.yaml` 설정 파일의 세부 내용은 다음과 같습니다.

```yaml
learning_rate: 0.001      # learning rate for model
dropout_rate: 0.1         # dropout rate for model
lambda_for_cl: 0.1        # control loss for encoder model(range: 0 ~ 1, allow float type)
substitution_rate: 0.3    # ratio for item substitution in batch
masking_rate: 0.3         # ratio for item masking in batch
cropping_rate: 0.3        # ratio for item cropping in batch
batch_size: 256       
sequence_len: 1           # if run_mode == 'diversity' you should use 1, else using 30 will be good 
embedding_dim: 128        # embedding size
early_stop_patience: 20   # control early stopping steps
warmup_steps: 10000       # control warmup steps for learning rate scheduling
epochs: 200              
clip_norm: 5.0            # gradient clipping norm
log_interval: 10          # interval for logging for loguru and wandb
temperature: 0.1          # temperature for NT-Xent Loss

try_num: '9'              # experiment number
run_mode: 'test'          # running mode for main.py: ['train', 'test', 'diversity']
data_type: 'RentTheRunway'  # 'RentTheRunway'
train_type: 'NCF'         # Model which you want to train: ['CL', 'REC', 'ONLY_REC', 'NCF']
aug_mode: 'substitute'    # Item augmentaion mode: ['substitute', 'crop', 'mask']
rec_num_dense_layer: 2
model_trainable: False    # True, False

k: 20                     # top-K item
sparsity: 0.5             # transaction count sparsity for checking diversity
```

### Experiment log

본 논문 실험 결과의 상세한 로그(파라미터 설정, 학습 또는 테스트 결과, 파라미터 시각화 등)는 아래 홈페이지에서 확인하실 수 있습니다.

#### [ISCLRec Weight & Bias logging](https://wandb.ai/jihoahn9303/KHU_graduate)

### Package

본 프로젝트를 위한 패키지는 `requirements.txt` 파일을 통해 다운로드 하실 수 있습니다.
