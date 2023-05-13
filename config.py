import transformers

DEVICE = "cuda"
num_classes = 1
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 25
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "./model.pth"
TRAINING_FILE = "./IMDB_Dataset.csv"
return_dict = False
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=False)