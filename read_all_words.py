with open('common_words.txt', 'r') as f:
	all_words = f.read().split()

print len(all_words)
print all_words[0]



# print len(all_words)
# awl = []
# for w in all_words:
# 	awl.append(w.lower())

# print awl[0]

with open('common_words_2.txt','w') as f:
	for w in all_words:
		f.write(w.lower()+"\n")

#with open('all_words_lowercase.txt','r') as f:
#	all_words=f.readlines()
#	print len(all_words)

# arr = [0,1,3]

# for i in arr[0:3]:
# 	print i