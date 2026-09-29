import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


class BERTWordEncoder(nn.Module):
    def __init__(
        self,
        bert_model_path="bert-large-uncased",
        h=8,
        initialize_pretrained=True,
    ):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_path, local_files_only=True
        )
        if initialize_pretrained:
            self.bert = BertModel.from_pretrained(
                bert_model_path, local_files_only=True
            )
        else:
            self.bert = BertModel(
                BertConfig.from_pretrained(bert_model_path, local_files_only=True)
            )
        self.h = h

    def forward(self, data_dict):
        text_list = data_dict["text"]
        tokens = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )

        tokens = {k: v.to(self.bert.device) for k, v in tokens.items()}
        bert_out = self.bert(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            return_dict=True,
        )

        seq = bert_out.last_hidden_state
        B, L, _ = seq.shape
        padding_mask = tokens["attention_mask"] == 0
        attention_mask = padding_mask[:, None, None, :]

        attention_mask = attention_mask.expand(B, self.h, L, L)
        adaptive_input = {"hidden_states": seq, "attention_mask": attention_mask}
        return adaptive_input
