#importing libraries
import torch
import streamlit as st
from transformers import BitsAndBytesConfig
from transformers import pipeline
import requests
from PIL import Image
# import matplotlib as plt is not supported
from io import BytesIO


st.title('DSG Image KG')

device="cuda" if torch.cuda.is_available() else "cpu"
st.text(f"Processing unit being used: {device}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

st.text("This is a deployment test")

#loading 4bit quantised LLav model
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

#to use passing any image path or image link to LOAD_IMAGE
def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

image = load_image("/content/action_genome_dataset1.jpg")
st.image(image, caption="Loaded Image", use_column_width=True)

#gen of VLM's text description of image
max_new_tokens = 200
prompt = "USER: <image>\nDescribe in detail the objects and what is hapenning in the image to me. The description should be of 200 words. \nASSISTANT:"
result_list=[]

for i in range(1):
  outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

  response_text = outputs[0]['generated_text']
  assistant_prefix = "ASSISTANT:"
  response = response_text.split(assistant_prefix, 1)[-1].strip() if assistant_prefix in response_text else response_text
    # Split the text by \n and .
  segments = response.split('.')
  lines = [segment.split('\n') for segment in segments]

  # Flatten the list of lists into a single list
  print(lines)

  for texts in lines:
    if texts[0] !='':
     sentence=texts[0]
     result_list.append(sentence)

  for sentence in result_list:
    st.text(sentence)

  #response = [item.strip() for sublist in lines for item in sublist if item.strip()]


st.text("This is a deployment test")