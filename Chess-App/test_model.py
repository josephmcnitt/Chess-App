import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the saved model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("trained_model.pt"))

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the input file
input_file = "output.txt"

# Define the output file
output_file = "predictions.txt"

# Open the input and output files
with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        if "FEN: " not in line:
            continue
        fen = line.split("FEN: ")[1].split(";")[0]

        # Tokenize the FEN string
        inputs = tokenizer(fen, padding=True, truncation=True, return_tensors='pt')

        # Make predictions with the model
        outputs = model(**inputs)
        probs = torch.softmax(outputs[0], dim=0)
        pred_label = torch.argmax(probs).item()

        # Write the predicted label to the output file
        f_out.write(fen + " " + str(pred_label) + "\n")

