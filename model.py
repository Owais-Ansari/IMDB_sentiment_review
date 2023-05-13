import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self,num_classes):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, return_dict = config.return_dict)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
 #BertModel.from_pretrained("bert-base-cased", return_dict=False)