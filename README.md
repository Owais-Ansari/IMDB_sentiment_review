# bert-sentiment movie review
requirements


python>=3.8 \n
torch==1.9.1+cu102 \n
torchaudio==0.9.1 \n
transformers==4.25.1 \n
matplotlib==3.5.1 \n
textblob==0.17.1 \n
emoji==2.0.0 \n
nlpaug==1.1.11 \n
sckit-learn==1.0.1 \n



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
