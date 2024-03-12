from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch
model_path = 'C:/Users/Dell/Downloads/GPT2-FineTune'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set the device (CPU or GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Example text input
input_text = "உங்களுக்கு என்ன சொல்ல விரும்புகிறீர்கள்?"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate text using the fine-tuned model
output = model.generate(input_ids, max_length=150, num_beams=5, temperature=1.0, no_repeat_ngram_size=2)

# Decode the generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print("Generated Text:", generated_text)