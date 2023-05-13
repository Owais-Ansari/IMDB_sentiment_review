import config
import torch
from utils import text_ip
from utils import nlp_aug
import numpy as np

class BERTDataset:
    def __init__(self, review, target, aug=False):
        self.review = review
        self.target = target
        self.aug = aug
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        
        review = str(self.review[item])
        review = text_ip.convert_lowercase(review)
        review = text_ip.remove_html_tags(review)
        #review = text_ip.remove_punc(review)
        reveiw = text_ip.remove_special_char(review)
        reveiw = text_ip.spelling_correct(review)
        reveiw = text_ip.remove_stopwords(reveiw)
        #reveiw = text_ip.stemming(review)
        if self.aug:
            rand_num = np.random.randint(10)
            if rand_num < 3: 
                review = nlp_aug.synonym_aug(reveiw)[0]
                review = nlp_aug.swap_words(review)[0]
            #elif rand_num >=3 & rand_num <=5:
            #    review = nlp_aug.summarize(review)[0] #nlp_aug.crop_sent
            elif rand_num >5 & rand_num <=8:
                review = nlp_aug.crop_sent(review)[0]
                #review = nlp_aug.replaces_by_context_word(review)[0]
            #else:
            #review = nlp_aug.back_translation(review)
            review = nlp_aug.insert_word(review)[0]
            review = nlp_aug.delete_random_char(review)[0]
        review = " ".join(review.split())    
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
            }
        
        
        
