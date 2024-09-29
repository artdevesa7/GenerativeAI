from transformers import AutoTokenizer, AutoModelForCausalLM

# Load pre-trained model and tokenizer from Hugging Face
model_name = "gpt2"  # You can change this to any model available on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Encode the input text
input_text = "tell me a joke"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
