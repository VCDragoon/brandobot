"""Core Flask app routes."""
from flask import render_template
from flask import current_app as app
import os
import time
import re
import random
import numpy as np
import pickle
import tensorflow as tf
import os
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse

import random

from csv import writer
  
#---------------------------------------------------------------------------------------#
# CSV Helper Function
#---------------------------------------------------------------------------------------#
def append_list_as_row(file_name, list_of_elem):
	# Open file in append mode
	with open(file_name, 'a+', newline='') as write_obj:
		# Create a writer object from csv module
		csv_writer = writer(write_obj)
		# Add contents of list as last row in the csv file
		csv_writer.writerow(list_of_elem)
#---------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------#
# Model Initialization/Load
#---------------------------------------------------------------------------------------#

# Model Seed
seed = 31
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
							
# Model Configs:
gpt2_small_config = GPT2Config()
gpt2_medium_config = GPT2Config(n_ctx=1024, n_embd=1024, n_layer=24, n_head=16)
gpt2_large_config = GPT2Config(n_ctx=1024, n_embd=1280, n_layer=36, n_head=20)  

model_size = "medium"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Model Loads:
if model_size == "small_custom":
	model = GPT2LMHeadModel(gpt2_small_config)
	model.load_state_dict(torch.load("custom/pytorch_model.bin"), strict=False)
elif model_size == "medium":
	model = GPT2LMHeadModel(gpt2_medium_config)
	model.load_state_dict(torch.load("medium_ft.pkl"), strict=False)
elif model_size == "large":
	model = GPT2LMHeadModel(gpt2_large_config)
	model.load_state_dict(torch.load("large_ft.pkl"), strict=False)
elif model_size == "small":
	model = GPT2LMHeadModel(gpt2_small_config)
	model.load_state_dict(torch.load("small_ft.pkl"), strict=False)

device = torch.device("cuda")
model = model.to(device)
model.lm_head.weight.data = model.transformer.wte.weight.data

# weights = torch.load('medium_ft.pkl')
# medium_config = GPT2Config(n_embd=1024,n_layer=24,n_head=16)
# large_config = GPT2Config(n_ctx=1024, n_embd=1280, n_layer=36, n_head=20) 

# model = GPT2LMHeadModel(medium_config)
# weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
# weights.pop("lm_head.decoder.weight",None)
# model.load_state_dict(weights)
# model.eval()
# model.to('cuda')
# torch.manual_seed(18)
# np.random.seed(18)
#---------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------#
# Non-configurable Variable Initialization
#---------------------------------------------------------------------------------------#
conditioned_tokens = []
generated_tokens = []
generated_tokens = []
#---------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------#
# Configurable Variable Initialization
#---------------------------------------------------------------------------------------#
save_chats = ["User Input", "BrandoBot Response"]
temperature = 0.9
print("Routes v51")
history = []
historyOn = True
num_history_messages = 5
	
therapist = ['As a therapist', 'Psychologically', 'Counseling Session', 'projecting emotions', 'I would advise you to' 
				'work on mental health', 'very worried about mental well being', 'work through these emotions', 'never hurt yourself']

jerk = ['you suck', 'f off', 'bitch', 'fatty', 'bite me', 'screw you']

noMemory = []
personality = []
personalityTrigger = 10   
#---------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------#
# Prediction API Route
#---------------------------------------------------------------------------------------#
@app.route('/prediction', methods=['GET', 'POST'])
def web_reply():
	userText = request.args.get('userText')
	if request.args.get('historyOn') =="True":
		historyToggle = True
	else:
		historyToggle= False
	historyLength = int(request.args.get('historyLength'))
	temperature = float(request.args.get('temperature'))
	top_k = int(request.args.get('top_k'))

	if request.args.get('randomHistory') == "False":
		randomHistory = random.randint(0, historyLength)
		print("random history: ", randomHistory)
	else:
		randomHistory = 0

	if request.args.get('personalityType') == "Jerk":
		personality = jerk
	elif request.args.get('personalityType') == "Therapist":
		personality = therapist
	else:
		personality = []

	#if len(userText) < 10:
	#	temperature = .25
	#	top_k = 20
	#elif len(userText) < 25:
	#	temperature = .4
	#	top_k = 30
	#elif len(userText) < 50:
	#	temperature = .65
	#	top_k = 45
	#else:
	#	temperature = 1
	#	top_k = 150

	#currentHistory = len(history)
	print("History Toggle:", historyToggle)
	print("History Length:", historyLength) 
	print("Temp: ", temperature)
	print("Top K: ", top_k)
	print("Personality Type: ", request.args.get('personalityType'))

	response = prediction(userText, historyToggle, historyLength, temperature, top_k, randomHistory, personality)
	return response
#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#
# Prediction Main Function
#---------------------------------------------------------------------------------------#
""" 
TODO: Embed the personality types within the SCORING (e.g., add to the score for selected response based on "therapy-type" category score
"""
def prediction(userText, historyToggle, historyLength, temperature, top_k, randomHistory, personality):

	if userText.lower() == "reset":
		response = "Memory cleared. Ready to begin anew."
		while len(history) > 0:
				del history[0]
		return response
	
	print("User Says: ", userText)
	#print(evaluateInput(encoder, decoder, searcher, voc, str(userText)))
	historyText = ""
	if historyToggle==True:
		while len(history) > historyLength:
			print("HISTORY BUFFER FULL: ", len(history), " |||| removing record: ", history[randomHistory])
			del history[randomHistory]
			randomHistory = random.randint(0, historyLength)
	else:
		while len(history) > 0:
			del history[0]
	for i in history:
		historyText = historyText + " " + i

		
	# see if the input string is longer than 4 characters, if so, embed personality response
	'''
	TODO: Need to redo this whole freakin thing... so messy
	'''
	tempString = ""
	if len(userText) > personalityTrigger and personality:
		tempString = random.choice(personality)
		print("Personality Buffer: " + tempString)
	if personality and historyToggle:
		print("String going to AI: " + historyText + " " + tempString + " " + userText)
		finalInputString = historyText + " " + tempString + " " + userText
		history.append(userText)
		history.append(tempString)
		
	elif not personality and historyToggle:
		print("String going to AI: " + historyText + " " + " " + userText)
		finalInputString = historyText + " " + " " + userText
		history.append(userText)
	else:
		print("String going to AI: " + userText)
		finalInputString = userText

	conditioned_tokens, generated_tokens = encode_text(finalInputString)
	
	response = decode_text(conditioned_tokens, generated_tokens, temperature, top_k)
	history.append(response)
	"""
	TODO: refactor the "save chats" into a function so I don't have to edit it in every block
	"""
	addChats = [userText, response]
	#save_chats.append(userText, response)
	
	print("BrandoBot Says: ", response)
	print(" ")

	append_list_as_row('Saved_Chats.csv', addChats)

	return response
#---------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------#
# Encoder - Tokenizer 
#---------------------------------------------------------------------------------------#
def encode_text(text):
	generated_tokens = []
	conditioned_tokens = []
	conditioned_tokens = tokenizer.encode(text) + [50256]
	return conditioned_tokens, generated_tokens
#---------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------#
# Decoder - Token Selection
#---------------------------------------------------------------------------------------#
def decode_text(conditioned_tokens, generated_tokens, temperature, top_k):
	while True:

		# for segment display purpose, keep 2 sets of tokens
		indexed_tokens = conditioned_tokens + generated_tokens
		tokens_tensor = torch.tensor([indexed_tokens])
		tokens_tensor = tokens_tensor.to('cuda')
		
		with torch.no_grad():
			outputs = model(tokens_tensor)
			predictions = outputs[0]

		logits = predictions[0, -1, :] / temperature
		filtered_logits = top_k_top_p_filtering(logits, top_k)
		
		probabilities = F.softmax(filtered_logits, dim=-1)

		### multinominal vs argmax probabilities
		next_token = torch.multinomial(probabilities, 1)
		#next_token = torch.argmax(probabilities, -1).unsqueeze(0)

		generated_tokens.append(next_token.item())
		result = next_token.item()

		if result == 50256:
			returnText =(tokenizer.decode(generated_tokens[:-1]))
			conditioned_tokens += generated_tokens
			generated_tokens = []
			return returnText
#---------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------#
# Decoder - Logit Selection
#---------------------------------------------------------------------------------------#
def top_k_top_p_filtering(logits, top_k, top_p=1, filter_value=-float('Inf')):
	""" Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
		Args:
			logits: logits distribution shape (vocabulary size)
			top_k >0: keep only top k tokens with highest probability (top-k filtering).
			top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
				Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
	"""
	assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
	top_k = min(top_k, logits.size(-1))  # Safety check
	if top_k > 0:
		# Remove all tokens with a probability less than the last token of the top-k
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = filter_value

	if top_p > 0.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		# Remove tokens with cumulative probability above the threshold
		sorted_indices_to_remove = cumulative_probs > top_p
		# Shift the indices to the right to keep also the first token above the threshold
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		indices_to_remove = sorted_indices[sorted_indices_to_remove]
		logits[indices_to_remove] = filter_value
	return logits
#---------------------------------------------------------------------------------------#



#---------------------------------------------------------------------------------------#
# Home/Cover Page
#---------------------------------------------------------------------------------------#
@app.route('/')
def home():
	"""Landing page."""
	return render_template(
		'index.jinja2',
		title='BrandoBot Entryway',
		description='BrandoBot.  Always watching, always waiting.',
		template='home-template',
		body="This is a homepage served with Flask."
	)
#---------------------------------------------------------------------------------------#
