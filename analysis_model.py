import sys
sys.path.append('C:\\users\\jmmag\\appdata\\local\\programs\\python\\python310\\lib\\site-packages')
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

with open('sample.pgn') as f:
    positions = [line.strip() for line in f]

class FENTransformer(nn.Module):
    def __init__(self, num_labels=2, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1):
        super(FENTransformer, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FENTransformer(num_labels=2)
# Tokenize the positions
inputs = tokenizer(positions, padding=True, truncation=True, return_tensors='pt')

# Make predictions with the model
outputs = model(**inputs)

# Extract the predicted probabilities
probs = torch.softmax(outputs.logits, dim=1)
loss, logits = outputs[:2]

print(loss)
print(logits)
