# -*- coding: utf-8 -*-
"""
### Imports ###
"""

!pip install openai

"""### Chat ###"""

from openai import OpenAI

api_key = "<Insert API Key here>"

def chat_with_gpt(prompt):
  client = OpenAI(api_key=api_key)
  response = client.chat.completions.create(
      model = 'gpt-3.5-turbo',
      messages = [
          {'role': 'user', 'content': prompt}
      ]
  )
  return response.choices[0].message.content

def open_chat():
  while True:
    user_input = input("You: ")
    if user_input.lower().strip in ['exit', 'done', 'bye', 'quit']:
      return 'Chat closed'
    response = chat_with_gpt(user_input)
    print("GPT3.5:", response)

open_chat()
