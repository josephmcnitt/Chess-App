import sys
sys.path.append('C:\\users\\jmmag\\appdata\\local\\programs\\python\\python310\\lib\\site-packages')
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader



with open('output.txt') as f:
    max_line_num = 10000  # maximum line number to include
    positions = []
    for i, line in enumerate(f):
        if i < max_line_num:
            positions.append(line.strip())


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
batch_size = 10

# Create dataset from positions and tokenize them
inputs = tokenizer(positions, padding=True, truncation=True, return_tensors='pt')
dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

# Create data loader with specified batch size
data_loader = DataLoader(dataset, batch_size=batch_size)

import torch.optim as optim

# Define optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Define number of training epochs
epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in data_loader:
        # Load batch to device
        input_ids, attention_mask, token_type_ids = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)

        # Set labels
        labels = torch.zeros(batch_size).long().to(device)

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass
        loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    # Print loss for epoch
    print(f"Epoch {epoch+1} loss: {running_loss/len(data_loader)}")

# Save trained model
torch.save(model.state_dict(), 'trained_model.pt')



