## Introduction

The contest will provide participants with a set of user ad click history records in a time window of 91 days (3 months) as a training data set. Each record contains the date (from 1 to 91), user information (age, gender), and information about the clicked advertisement (material id, advertisement id, product id, product category id, advertiser id, advertiser industry) id, etc.), and the number of times the user clicked on the ad that day. The test data set will be the ad click history of another group of users. The test data set provided to the contestants will not contain the age and gender information of these users. This question requires participants to predict the age and gender of the users appearing in the test data set.

### 1. Environment

- Pytorch
- Linux Ubuntu 16.04, 256G内存，4*p100
- pip install transformers==2.8.0 pandas gensim scikit-learn filelock gdown

### 2. Model

![avatar](picture/model.png)
![avatar](picture/mlm.png)
![avatar](picture/fusion-layer.png)
![avatar](picture/output.png)

### 3. Limited Resources Scripts

1) If there is insufficient memory or you just want to simply run the complete code, please use only the preliminary data:

src/prepocess.py 8, 15, 22

2) bert-small and adjust batch size

### 4. To run the complete code

You can run the following script to run the entire process and generate results. Or follow the instructions in section 3-7 to run in sequence

```shell
bash run.sh
```

### 5. Data Download

https://drive.google.com/file/d/15onAobxlim_uRUNWSMQuK6VxDsmGTtp4/view?usp=sharing

You can get the data by running the following shell script

```shell
gdown https://drive.google.com/uc?id=15onAobxlim_uRUNWSMQuK6VxDsmGTtp4
unzip data.zip 
rm data.zip
```

### 6. Data preprocessing

Combine all files and divide them into click record files (click.pkl) and user files (train_user.pkl/test_user.pkl)

```
python src/preprocess.py
```

### 7. Feature Extraction

```shell
python src/extract_features.py
```

### 8. Pre-training Word2Vector and BERT

There are two ways to obtain pre-training weights: re-training or downloading pre-trained weights

Note: The weights of Word2Vector and BERT must be the same, that is, either all re-training or downloading all

#### 1) Pre-training Word2Vector

Pre-trained word2vector 

```shell
python src/w2v.py
```

Or download the pre-trained [W2V](https://drive.google.com/file/d/1SUpukAeXR5Ymyf3wH3SRNdQ3Hl2HazQa/view?usp=sharing)

```shell
gdown https://drive.google.com/uc?id=1SUpukAeXR5Ymyf3wH3SRNdQ3Hl2HazQa
unzip w2v.zip 
cp w2v/* data/
rm -r w2v*
```

#### 2) Pre-training BERT

Pre-training BERT (if the GPU is v100, you can install apex and add --fp16 to the parameters for acceleration) 

```shell
cd BERT
mkdir saved_models
python run.py \
    --output_dir saved_models \
    --model_type roberta \
    --config_name roberta-base \
    --mlm \
    --block_size 128 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 1.0 \
    --max_steps 100000 \
    --mlm_probability 0.2 \
    --warmup_steps 10000 \
    --logging_steps 50 \
    --save_steps 10000 \
    --evaluate_during_training \
    --save_total_limit 500 \
    --seed 123456 \
    --tensorboard_dir saved_models/tensorboard_logs    
rm -r saved_models/bert-base    
cp -r saved_models/checkpoint-last saved_models/bert-base
rm saved_models/bert-base/optimizer.pt
cp saved_models/vocab.pkl saved_models/bert-base/vocab.pkl
cd ..
```

Or download the pre-trained [BERT-base](https://drive.google.com/file/d/1ToAJwl_oRAeRNyYF_FK0B2APVXlPFTlq/view?usp=sharing) 

```shell
gdown https://drive.google.com/uc?id=1ToAJwl_oRAeRNyYF_FK0B2APVXlPFTlq
unzip bert-base.zip
mv bert-base BERT/
rm bert-base.zip
```

### 9. Model Training

```shell
mkdir saved_models
mkdir saved_models/log
for((i=0;i<5;i++));  
do  
  python run.py \
      --kfold=5 \
      --index=$i \
      --train_batch_size=256 \
      --eval_steps=5000 \
      --max_len_text=128 \
      --epoch=5 \
      --lr=1e-4 \
      --output_path=saved_models \
      --pretrained_model_path=BERT/bert-base \
      --eval_batch_size=512 2>&1 | tee saved_models/log/$i.txt
done  
```
Combine the results, the result is submission.csv

```shell
python src/merge_submission.py
```

### 10. Pre-trained models of different scales

Since this competition incorporates pre-training models of different scales, pre-training models of different scales are also provided here:

[BERT-small](https://drive.google.com/file/d/1bDneO-YhBs5dx-9qC-WrBf3jUc_QCIYn/view?usp=sharing), [BERT-base](https://drive.google.com/file/d/1ToAJwl_oRAeRNyYF_FK0B2APVXlPFTlq/view?usp=sharing), [BERT-large](https://drive.google.com/file/d/1yQeh3O6E_98srPqTVwAnVbr1v-X0A7R-/view?usp=sharing), [BERT-xl](https://drive.google.com/file/d/1jViHtyljOJxxeOBmxn9tOZg_hmWOj0L2/view?usp=sharing)

Among them, bert-base works best

```shell
#bert-small
gdown https://drive.google.com/uc?id=1bDneO-YhBs5dx-9qC-WrBf3jUc_QCIYn
#bert-base
gdown https://drive.google.com/uc?id=1ToAJwl_oRAeRNyYF_FK0B2APVXlPFTlq
#bert-large
gdown https://drive.google.com/uc?id=1yQeh3O6E_98srPqTVwAnVbr1v-X0A7R-
#bert-xl
gdown https://drive.google.com/uc?id=1jViHtyljOJxxeOBmxn9tOZg_hmWOj0L2
```
