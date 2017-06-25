import csv
import itertools
import operator
import numpy as np
import nltk #Natural Language ToolKit
import sys
import time
from datetime import datetime
from tools import *


datafile = 'data/reddit-comments-2015-08.csv'
vocabulary_size = 8000
hidden_dim = 80
unknown_token = 'UNKOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'

# Read data and append tokens
print('Reading data')
t1 = time.time()
with open(datafile, newline='', encoding='utf-8') as f:
	#readFile = f.read()
	#readFile = readFile.decode('utf-8')
	#print(readFile[8000:8010])
	##readFile = readFile.split('\n')[1:]
	#print(readFile[0:100])
	#print('-'*100)
	reader = csv.reader(f, skipinitialspace=True)

	reader.__next__() #Skip initial string that just reads "body"
	sentences = itertools.chain(
		*[nltk.sent_tokenize(x[0].lower()) for x in reader])
	sentences = ['%s %s %s' % (sentence_start_token, x, sentence_end_token)
	 for x in sentences]
t2 = time.time()
print('Parsed %d sentences' % len(sentences))
print('Duration: %d ms' % ((t2-t1)*1000))

#Tokenize sentences into words
tokenized_sentences = [nltk.word_tokenize(s) for s in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print('Found %d unique word tokens.' % len(word_freq.items()))



## For every tokenized sentence replace words with unkown_token if 
## freq of word is not top 8000

vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print('The most frequent word in our vocabulary is "%s" and appeared %d times.'
	% (vocab[3][0], vocab[3][1]))
print('The least frequent word in our vocabulary is "%s" and appeared %d times.'
 % (vocab[-1][0], vocab[-1][1]))

for i, sent in enumerate(tokenized_sentences):
	tokenized_sentences[i] = [w if w in word_to_index else unknown_token
	 for w in sent]

print(type(tokenized_sentences))


##Save data
name = str(np.random.random())
name = name + str(np.random.random())
name = name + str(np.random.random())
filename = name + '.csv'
with open(filename, 'w', newline='', encoding='utf-8') as f:
	writer = csv.writer(f)
	for j, sent in enumerate(tokenized_sentences):
		writer.writerow(tokenized_sentences[j])
np.savez('wordtofromindex.txt', wtoi = word_to_index, itow = index_to_word)

