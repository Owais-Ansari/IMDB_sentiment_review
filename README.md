# Bert-sentiment movie review
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
## Dataset   
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download
## Dependencies


- python>=3.8 
- torch==1.9.1+cu102
- torchaudio==0.9.1 
- transformers==4.25.1 
- matplotlib==3.5.1 
- textblob==0.17.1 
- emoji==2.0.0 
- nlpaug==1.1.11 
- sckit-learn==1.0.1 



### Training

```shell script
> python train.py 

Update the config.py file

arguments:
	DEVICE = "cuda"  # cuda device 
	num_classes = 1  # 
	MAX_LEN = 512    #len of the postional vector
	TRAIN_BATCH_SIZE = 16
	VALID_BATCH_SIZE = 16
	EPOCHS = 25
	BERT_PATH = "bert-base-uncased"
	MODEL_PATH = "./model.pth"
	TRAINING_FILE = "./IMDB_Dataset.csv"
	return_dict = False
	TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=False)
	
	
```
