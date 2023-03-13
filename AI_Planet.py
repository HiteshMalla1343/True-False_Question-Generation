import streamlit as st

# Define a function to generate True/False question based on user input
import torch
from transformers import pipeline
import random
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

def get_question(answer, context, max_length=64):
  input_text = "answer: %s  context: %s </s>" % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])
def generate_question(context):
    options = ['True', 'False']
    answer=random.choice(options)
    statement =get_question(answer,context)    
    return statement, options, answer
    
st.title("True/False Question Generator")
context = st.text_input("Enter context:")
submit_button = st.button("Submit")
answer_submitted = None
correct_answer = None
user_answer=None
# Define behavior of UI elements
if submit_button:
    statement, options, answer = generate_question(context)
    st.write(statement)
    options=['True', 'False']
    user_answer = st.radio("Select an answer:", options)
if user_answer is not None:
    answer_submitted = True
if answer_submitted is not None:
    if user_answer == answer:
        correct_answer = True
    else:
        correct_answer = False

if correct_answer is not None:
    if correct_answer:
        st.write("Your answer is correct!")
    else:
        st.write("Your answer is incorrect.")
