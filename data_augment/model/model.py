import torch
import torch.nn as nn
from transformers import BertModel

class BertForParsing(nn.Module):
    def __init__(self, num_pos_labels, num_dep_labels, max_length=128, pretrained_model_name="bert-base-multilingual-cased"):
        super(BertForParsing, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.pos_classifier = nn.Linear(hidden_size, num_pos_labels)
        self.dep_classifier = nn.Linear(hidden_size, num_dep_labels)
        self.head_classifier = nn.Linear(hidden_size, max_length)
        self.max_length = max_length
    
    def forward(self, input_ids, attention_mask, pos_labels=None, dep_labels=None, head_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # shape: (batch, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        pos_logits = self.pos_classifier(sequence_output)
        dep_logits = self.dep_classifier(sequence_output)
        head_logits = self.head_classifier(sequence_output)
        loss = None
        if pos_labels is not None and dep_labels is not None and head_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            pos_loss = loss_fct(pos_logits.view(-1, pos_logits.size(-1)), pos_labels.view(-1))
            dep_loss = loss_fct(dep_logits.view(-1, dep_logits.size(-1)), dep_labels.view(-1))
            head_loss = loss_fct(head_logits.view(-1, head_logits.size(-1)), head_labels.view(-1))
            loss = pos_loss + dep_loss + head_loss
        return pos_logits, dep_logits, head_logits, loss

# sanity check
if __name__ == "__main__":
    import torch
    dummy_input_ids = torch.randint(0, 1000, (2, 16))
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    dummy_pos_labels = torch.randint(0, 5, (2, 16))
    dummy_dep_labels = torch.randint(0, 10, (2, 16))
    dummy_head_labels = torch.randint(0, 16, (2, 16))
    model = BertForParsing(num_pos_labels=5, num_dep_labels=10, max_length=16)
    pos_logits, dep_logits, head_logits, loss = model(dummy_input_ids, dummy_attention_mask,
                                                      dummy_pos_labels, dummy_dep_labels, dummy_head_labels)
    print("POS logits shape:", pos_logits.shape)
    print("Dep logits shape:", dep_logits.shape)
    print("Head logits shape:", head_logits.shape)
    print("Loss:", loss.item())
