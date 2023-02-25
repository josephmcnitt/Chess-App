import sys
sys.path.append('C:\\users\\jmmag\\appdata\\local\\programs\\python\\python310\\lib\\site-packages')
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader



with open('output.txt') as f:
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
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FENTransformer(num_labels=2)

# Set batch size
batch_size = 2

# Create dataset from positions and tokenize them
inputs = tokenizer(positions, padding=True, truncation=True, return_tensors='pt')
dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

# Create data loader with specified batch size
data_loader = DataLoader(dataset, batch_size=batch_size)

# Make predictions with the model
outputs = model(**inputs)

# Extract the predicted probabilities
probs = torch.softmax(outputs[0], dim=0)
loss, logits = outputs[1:]

output_file = "output_guesses.txt"

with open(output_file, "w") as f_out:
    f_out.write(str(outputs))
    f_out.write(str(probs))
    f_out.write(str(loss))
    f_out.write(str(logits))

