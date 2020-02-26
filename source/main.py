#first part: getting text
#https://machinetalk.org/2019/02/01/text-generation-with-tensorflow/
#Part 2: Creating the model
#https://machinetalk.org/2019/02/08/text-generation-with-pytorch/

##Torch Library
import torch
import torch.nn as nn
##Torch Library

#To deepcopy
import copy
#Around float numbers, faster method there I find
from math import floor
#To check time
import time
#Numpy to use math
import numpy as np
#Generating the dictionary
from collections import Counter
#Files help
import glob
#Argmentation
import argparse

#--- ini archives ---
###   BATCH_SIZE   ###
#Number of lists those will feed the net.
##ex: 1 batch and 3 seq_size:
#(array([[22, 10, 59]]), array([[10, 59, 23]])) 
##ex: 2 batch and 3 seq_size:
#(array([[11,  2,  3],[22, 10, 59]]), array([[ 2,  3,  0],[10, 59, 23]]))
#batch_size = 5

#Higher seq_size is, more phrase length can be translated to nn
#number of itens inside of each list.
##ex: 3 batch and 1 seq_size:
#(array([[45],[11],[7]]),array([[17],[2],[2]]))
##ex: 3 batch and 2 seq_size:
#(array([[16,45],[1,11],[6,7]]),array([[45,17],[11,2],[7,2]]))
##ex: 3 batch and 5 seq_size:
#(array([[9,41,42,43,9],[49,24,17,18,50],[58,21,22,10,59]]),array([[41,42,43,9,44],[24,17,18,50,24],[21,22,10,59,2]]))
#seq_size   = 11

#
#embedding = 64

#64
#lstm_size=128

#Number of times that all selected archives will be read
#epochs = 999999999

#Number of chars generated to output
#out_char = 100

#Twins
#False - is not aceptable same words
#True  - is aceptable same words.
#twins = True

#5
#gradients_norm=5

#Directories name
#_dir = 'goncalves_dias'
#_dir = 'got'

#Input words. 
#ini_words = 'Minha terra nao tem'
#ini_words = 'Daenerys'
#Input words as array
##ini_words_v = []
##for w in ini_words.lower().split(" "):  ini_words_v.append( w )


#argmuntation
#Call example:  python main.py -d 'directory' -i 'frase' 
def args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', type=str,  required=False, default='goncalves_dias', help='Name of dir')
	parser.add_argument('-i', type=str,  required=False, default='Minha terra',    help='Ini words')
	parser.add_argument('-n', type=int,  required=False, default=5,                help='Normalized gradient value')
	parser.add_argument('-t', type=bool, required=False, default=True,             help='Same words are allowed by default. Utilize -t False to disable that')
	parser.add_argument('-e', type=int,  required=False, default=1000,             help='Number of times tht all texts will be read')
	parser.add_argument('-o', type=int,  required=False, default=100,              help='Number of chars predicted each 100 iterations. Default: 100')
	parser.add_argument('-b', type=int,  required=False, default=5,                help='Number of words utilized in each batch')
	parser.add_argument('-z', type=int,  required=False, default=-1,              help='Number of iterations qhen the model will be saved to future use')
	parser.add_argument('-s', type=int,  required=False, default=11,               help='Number of words utilized in learning process. Higher this number, deeper learning')
	#
	return parser.parse_args()

#--Argmentation
arg = args()
batch_size     = arg.b
seq_size       = arg.s
embedding      = 64
lstm_size      = 128
epochs         = arg.e
out_char       = arg.o
twins          = arg.t
gradients_norm = arg.n
_dir           = arg.d
ini_words      = arg.i
#ini_words = 'Daenerys'
#Input words as array
ini_words_v = []
for w in ini_words.lower().split(" "):  ini_words_v.append( w )
#--Argmentation

#reading text content
#return:
# int_to_voc - Dictionary: {indice:'word'}
# voc_to_int - Dictionary: {'word', indice}
# max_words  - Number of differet words in db
# in_text    - All sample data --- 3/36
# out_text   - Answer data --- 3/36
#twins       - False-not allowed same words. True-allow same words
def get_data(ini_words_v, batch_size, seq_size, twins=False):
	if twins==False: print('|Not aceptable same words')
	else: print ('|Aceptable same words')
	#open all .txt documents in db dir:
	allDocs = glob.glob( '../data/'+_dir+'/*.txt' )
	alltext = []
	print ('  |-- Adding archives:')
	for doc in allDocs:
		print ('    |-- Loading '+doc)
		#full represents all text file content
		with open(doc, encoding='utf-8') as full:
			#text represents "full" readeable
			#lower, to dont re-match with upset words.
			text = full.read().lower()
			#now, text is an array of words, separated by espace
			text = text.split()

		for word in text:
			if twins==False and word in alltext:
				continue
			alltext.append(word)
		print ('    |-- Loaded  '+doc)
	#Putting ini words to main dictionary, but only those what doesnt already
	for w in ini_words_v:
		#Only add, if this word doenst belloong to dictionary:
		if twins==False and w in alltext:
			continue
		alltext.append(w)
	#Put all text in alltext:
	
	#word_count = represents a dictionary with all diferent words inside of archive. ex: 
	#{'bird':30, 'man':100, ...}
	word_count = Counter( alltext )
	
	#Another dictionary, but now with sorted content:
	sort = sorted(word_count, key=word_count.get, reverse=True)
	#Each word now corresponds in one unique counter. From 0 to max:
	#indice: number from 0 to max. Increments each loop
	#word: the word iterable this time.
	#---int_to_voc[0] - first word of list
	int_to_voc  = { indice:word for indice, word in enumerate(sort) }
	#Now, the reverse:
	#---voc_to_int['Sabiá;] - indice qhere has this word'
	#int_to_voc = { k:w for k, w in enumerate(sorted) }
	voc_to_int = { word:indice for indice, word in enumerate(sort) }
	#Number of words possible to learn:
	max_words = len( int_to_voc )
	
	#Array that reresents indices from each word of text
	text_in_indices = [voc_to_int[w] for w in alltext]
	#Number of baches
	_batches = int ( len(text_in_indices)/(seq_size* batch_size) )
	#Array that represents the text input. 
	in_text = text_in_indices[ :_batches * batch_size * seq_size]
	#Array with zeros that works like a mask to output
	out_text = np.zeros_like(in_text)
	#Answers are always the next word, so we need to move the indices
	#out_text[:-1] - all array but not the last one
	#in_text[:1]   - all array from position 1 
	out_text[:-1] = in_text[1:]
	#Now, the out_text last position will be the first position of in_text
	out_text[-1] = in_text[0]
	#To ensure that array has only one dimension and their sizes are equal each one 
	in_text = np.reshape(in_text, (batch_size, -1) )
	out_text = np.reshape(out_text, (batch_size, -1) )
	return int_to_voc, voc_to_int, max_words, in_text, out_text
	
#Generates inputs to feed network
def get_batches(in_text, out_text, b_size, seq_size):
	#np.prod = multiplication between shapes in an int
	n_batch = np.prod(in_text.shape) // ( seq_size * batch_size)
	#Generates the input to feed the network. Is a generator, that could be asseced by list(var)[x], where inside of it theres a tuple: left side: input and right side: answer.
	##ex:
	#(array([[45],[11],[7]]),array([[17],[2],[2]]))
	for i in range( 0, n_batch * seq_size, seq_size ):
		yield ( in_text[:, i:i+seq_size], out_text[:, i:i+seq_size] )

#Network
class NL(nn.Module):
	def __init__ ( self, max_words, seq_size, embedding_size, lstm_size ):
		super(NL, self).__init__()
		self.seq_size = seq_size
		self.lstm_size = lstm_size
		#2D array. 
		#Rows = Different words in dictionary
		#Cols = Embeding size
		self.embedding = nn.Embedding( max_words, embedding_size )

		self.lstm = nn.LSTM(
			embedding_size,
			lstm_size,
			batch_first=True
		)
		self.dense = nn.Linear( lstm_size, max_words )
	
	#x - ndarray int
	def forward( self, x, prev_state ):
		embed = self.embedding(x)
		output, state = self.lstm(embed, prev_state)
		logits = self.dense(output)
		return logits, state

	#Return two masks to use in 
	def zero(self, batch_size):
		return ( 
			torch.zeros(1, batch_size, self.lstm_size),
			torch.zeros(1, batch_size, self.lstm_size)
		)

def getLoss(net, lr=0.001):
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	return criterion, optimizer
	
def predict(device, net, ini_words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
	#Stop changing nn weights, to utilize validation method
	net.eval()

	state_h, state_c = net.zero(1)
	
	#Loop that goes over initial words
	for w in ini_words:
		#x = list(vocab_to_int)[y]
		#error, word that doesnt belongs to dictionary. See your ini_word
		if w not in ini_words:
			print ('Error. Word doest not belong to dictionary')
			exit(1)
		#Long tensor, to match embedding.
		#ix corresponds to one word.
		ix = torch.tensor( [[ vocab_to_int[w] ]])
		ix = ix.type( torch.LongTensor )
		#Consider IX, as the previous state. Sending to forward function of net.
		#state_h, state_c)
		output, (state_h, state_c) = net(ix, (state_h, state_c))
	
	_, top_ix = torch.topk(output[0], k=top_k)
	#choice - Int number of word
	choices = top_ix.tolist()
	choice = np.random.choice(choices[0])

	l_ini_words = copy.deepcopy( ini_words )
	l_ini_words.append(int_to_vocab[choice])

	for _ in range( out_char ):
		ix = torch.tensor([[choice]])
		ix = ix.type( torch.LongTensor )
		
		output, (state_h, state_c) = net(ix, (state_h, state_c))

		_, top_ix = torch.topk(output[0], k=top_k)
		choices = top_ix.tolist()
		choice = np.random.choice(choices[0])
		l_ini_words.append(int_to_vocab[choice])

	print(' '.join(l_ini_words))

#Receive the initial time, number of current epochs and max number of epochs.
#returns a string corresponding to time in years, months, weeks, days, hours, minues and seconds
def time_rest(ini, epoch, epochs):
	#GEtting time passed until here
	_now = time.time()-ini
	#Leeting better to huge amount of texts. The real time will be get after 1 epoch
	if epoch==0: 
		rest= floor( epochs*_now )
	else:
		rest=floor( (epochs - epoch)*_now/epoch) 
	#R, represents the final string to see time.
	r = ''
	#Year
	if ( rest>= 29030400):
		r=r+''+str( floor(rest/29030400) )+'Years '
		rest = rest-floor(rest/29030400)*29030400
	#Months
	if ( rest>= 2419200 ):
		r=r+''+str( floor(rest/2419200) )+'Months '
		rest = rest-floor(rest/2419200)*2419200
	#Weeks
	if ( rest>= 604800 ):
		r=r+''+str( floor(rest/604800) )+'Weeks '
		rest = rest-floor(rest/604800)*604800
	#days
	if ( rest>= 86400):
		r=r+''+str( floor(rest/86400) )+'Days '
		rest = rest-floor(rest/86400)*86400
	#Hours
	if (rest >= 3600):
		r=r+''+str( floor(rest/3600) )+'Hrs '
		rest = rest-floor(rest/3600)*3600
	#Mins
	if (rest >= 60):
		r = r+''+str( floor(rest/60) )+'Min '
		rest = rest-floor(rest/60)*60
	#Seconds
	r = r+''+str( floor(rest) )+'Seg'
	#Returning 
	return r

#Main function
def main():	
	#Change this to use cuda (My gpu is old... so I cant use it.)
	device = torch.device('cpu')
	
	print ('|Loading DB')
	int_to_voc, voc_to_int, max_words, in_text, out_text = get_data(ini_words_v, batch_size, seq_size, twins)
	print ('|Dictionary composed by '+str(max_words)+' words')
	net = NL(max_words, seq_size, embedding, lstm_size)
	criterion, optimizer = getLoss(net, 0.01)
	
	it = 0
	print('|Starting Train')
	#Inseconds
	start_time = time.time()
	for epoch in range(epochs):
		#Get the new batch
		batch = get_batches( in_text,out_text,batch_size, seq_size )
		#Get two masks composed by zero
		state_h, state_c = net.zero( batch_size )
		#x - ndarray int, representing in_text from next batch
		#y - ndarray int, representing out_text from next batch
		for x,y in batch:
			#Number of iterations
			it +=1
			#transform in Long tensors to embeddings funtion inside train
			x,y = torch.from_numpy(x), torch.from_numpy(y)
			x,y = x.type( torch.LongTensor ), y.type( torch.LongTensor )
			#Train the network
			net.train()
			#Gradient to zero, before backpropagation
			optimizer.zero_grad()
			#logits -  a tensor var
			logits, (state_h, state_c) = net( x, (state_h, state_c) )
			#--
			loss = criterion(logits.transpose(1,2), y)
			#--
			state_h = state_h.detach()
			state_c = state_c.detach()
			#--
			loss_v = loss.item()
			#--
			loss.backward()
			#--
			_ = torch.nn.utils.clip_grad_norm_( net.parameters(), gradients_norm )
			#--
			optimizer.step()
			
			#Print
			if it %100 == 0:
				#remaining time
				rem = time_rest(start_time, epoch, epochs)
				print('  |-- Batches analised: {}'.format(it),'Loss: %.5f' %(loss_v ),'Epoch: {}/{}'.format(epoch, epochs), ' Remaining time: '+rem )
				#Generates the output in text
				predict(device, net, ini_words_v, max_words, voc_to_int, int_to_voc, top_k=5)
				print ('  |--')
			#save at each 500:
			if arg.z > 0 and it %arg.z == 0:
				torch.save( net.state_dict(), '../model/model_{}'.format(_dir) )
				
	
	#Save
	torch.save( net.state_dict(), '../model/model_{}'.format(_dir) )


if __name__ == '__main__':
	main()