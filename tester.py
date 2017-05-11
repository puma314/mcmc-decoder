import mcmc
import timeit
from time import time
import numpy as np

def eval_accuracy(perm, y, ground):
	assert(len(y) == len(ground))
	wrong = 0
	for i, ch in enumerate(y):
		convert = alphabet_map[perm[ch]]
		if convert != ground[i]:
			wrong += 1
 
	return 1.-float(wrong)/len(ground)

def accuracy(true, attempt):
	assert(len(true)==len(attempt))
	wrong = 0
	for t, a in zip(list(true),list(attempt)):
		if t != a:
			wrong += 1
	return 1.-float(wrong)/len(true)

plaintext_files = ['plaintext1.txt','plaintext2.txt','plaintext3.txt','plaintext4.txt','plaintext500.txt','plaintext200.txt']

for _ in range(40):
	for textfile in plaintext_files:
		with open(textfile,'r') as f:
			text = f.read().rstrip()

		plaintext = text #for final checking
		print 'plaintext', textfile, 'length', len(plaintext)

		cipher = np.random.permutation(28)
		text = list(text)
		num_text = [mcmc.alphabet_to_num[x] for x in text]
		ciphered_num_text = [cipher[n] for n in num_text]
		ciphered_text = [mcmc.num_to_alphabet[n] for n in ciphered_num_text]
		ciphered_text = ''.join(ciphered_text)
		
		reverse_cipher = np.argsort(cipher)

		start = time()
		best_perm, deciphered = mcmc.decode(ciphered_text)
		end = time()

		print type(deciphered)

		acc = accuracy(plaintext, deciphered)
		#print 'true decoder, guess decoder', reverse_cipher, best_perm
		print 'accuracy', acc
		print 'time', end-start
		with open('deciphered_'+textfile,'w') as f:
			f.write(deciphered)