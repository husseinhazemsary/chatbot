import streamlit as st
import json
from datetime import datetime
import torch
from transformers import pipeline
import asyncio

# Ensure proper event loop handling for compatibility
try:
    asyncio.set_event_loop(asyncio.new_event_loop())
except Exception:
    pass

# Show title and description.
st.title("AI Response Generator")
st.write("Enter a prompt and select a temperature to generate AI responses.")

# Detect GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a text generation model (ensure you have Hugging Face access if required)
generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1, truncation=True)

# Function to generate responses
def generate_responses(prompt, temperature, num_responses=5):
    responses = generator(prompt, max_length=200, num_return_sequences=num_responses, temperature=temperature)
    return [r["generated_text"] for r in responses]

# Function to save results
def save_results(prompt, temperature, responses):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{temperature}_{timestamp}.json"
    data = {
        "prompt": prompt,
        "temperature": temperature,
        "responses": responses
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# User input for prompt and temperature
prompt = st.text_area("Enter your prompt:")
temperature = st.slider("Select Temperature", 0.0, 2.0, 0.7, 0.1)
num_responses = st.number_input("Number of Responses", min_value=1, max_value=10, value=5)

if st.button("Generate Responses"):
    if prompt:
        responses = generate_responses(prompt, temperature, num_responses)
        save_results(prompt, temperature, responses)
        
        st.subheader("Generated Responses:")
        for i, response in enumerate(responses, 1):
            st.write(f"### Response {i}")
            st.write(response)
    else:
        st.warning("Please enter a prompt before generating responses.")
