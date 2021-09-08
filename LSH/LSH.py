import timeit
import random
import numpy as np
from nltk.tokenize import RegexpTokenizer
import math


class Shingler:
	'''
	This class contiains functions to create shingles and assign them a shingle ID
	It also creates a map that associates each shingle with its id
	'''	

	shingle_set={}
	shingle_ID=[]
	shingle_Map=[]	
	
	def __init__(self,plots):
		token_list=self.preProcess(plots)
		j=0
		self.shingle_set=self.makeKShingles(token_list)
		self.shingle_Map=self.makeShingleMap(self.shingle_set)
		self.shingle_ID = [j for j in range(len(self.shingle_set))]
		
		
	def preProcess(self,plots):
		'''
		This method preprocesses the texts by removing unwanted elements and tokenizing the text
		'''	
		tokenizer = RegexpTokenizer("[\w']+")
		for i in range(len(plots)):
			tokens = tokenizer.tokenize(plots[i])
			plots[i] = tokens
		return plots
		
	def makeKShingles(self,plots):
		'''
		This method makes shingles of length k_shingles and stores them in a set
		'''
		k_shingles = 5
		shingle_set = {}
		count=0
		for i in range(len(plots)):
			for j in range(len(plots[i])- k_shingles):
					shingle = plots[i][j:j+k_shingles]
					shingle = ' '.join(shingle)
					if shingle in shingle_set.keys():
						shingle_set[shingle][i] += 1
						count+=1						
					else:
						z = np.zeros(len(plots))
						z[i] = 1
						shingle_set[shingle] = z
		return shingle_set
      	
	def makeShingleMap(self,shingle_set):
		'''
		This method maps each shingle to an ID
		'''
		index=0
		shingle_Map = []
		for key in shingle_set.keys():
			shingle_Map.append([index,key])
			index+=1
		return shingle_Map



'''
import all the files from the dataset and make a list
then we store text from each file in the list
'''
start = timeit.default_timer()

d = []
x=201# no. of documents in the dataset
for i in range(1,x):
	d.append(open(str(i)+".txt").read())
plots = d

end = timeit.default_timer()
print('Import: ', end - start)

start = timeit.default_timer()
sh=Shingler(plots)

shingle_set=sh.shingle_set
shingle_ID=sh.shingle_ID
shingle_Map=sh.shingle_Map

end=timeit.default_timer()
print('Shingling and Shingle mapping: ', end - start)

'''
create a hashmap for our shingle_ID
'''
start=timeit.default_timer()
nhashes = 20
hash_funcs = []

for i in range(nhashes):
  x = random.sample(shingle_ID, len(shingle_ID))
  while x in hash_funcs:
    x = random.sample(shingle_ID, len(shingle_ID))
  hash_funcs.append(x)

end = timeit.default_timer()
print('Hashing: ', end - start)

'''
create a signature matrix for our shingle matrix
'''
start = timeit.default_timer()

signature = np.full(shape=(nhashes, len(plots)), fill_value=len(shingle_set)+2)
for i in range(nhashes):
  for id in shingle_ID:
    hash_id = hash_funcs[i][id]
    for x in range(len(plots)):
      if shingle_set[shingle_Map[id][1]][x]==1:
        if hash_id<signature[i][x]:
          signature[i][x] = hash_id

end = timeit.default_timer()
print('Signature Matrix: ', end - start)

def jaccard(c1,c2):
  """
  Calculates the Jaccard similarity of Columns C1 and C2
  """
  x=0
  y=0
  z=0
  for key in shingle_set.keys():
    if shingle_set[key][c1] == shingle_set[key][c2] and shingle_set[key][c1] == 1:
      x+=1
    elif shingle_set[key][c1] == 1:
      y+=1
    elif shingle_set[key][c2] == 1:
      z+=1
  
  return x/(x+y)

start = timeit.default_timer()
'''
We split the matrix into bands and create Buckets of bucket_size for LSH
'''

rows = 5
bands = math.floor(len(signature)/rows)
bucket_size = 2
buckets = np.zeros(shape=(bands, len(plots)))

for band in range(bands):
  for j in range(signature.shape[1]):
    b = 0
    for i in range(rows):
      b+=signature[band*rows + i][j]%bucket_size
    buckets[band][j]=b

end = timeit.default_timer()
print('Buckets and Bands: ', end - start)

# Check similiarity LSH
# Candidate Pairs for doc
'''
we take the doucment number as input and compare it to from candidate pairs 
we find all possible candidate pairs and print them
'''
doc = int(input("Enter a document ID "))
doc=doc-1
start = timeit.default_timer()
candidate_list = []
threshold = 3

for c in range(signature.shape[1]):
  counter = 0
  for band in range(bands):
    if c !=doc:
      if buckets[band][c] == buckets[band][doc]:
        counter+=1
  if counter>=threshold:
    candidate_list.append(c)
    
print("The candidate pairs are-")
for c in candidate_list:
  print(str(doc+1)+","+str(c+1))

j = [(jaccard(doc, c), c) for c in candidate_list]
j.sort(reverse=True)
'''
we calculate the jaccard similarity for all the candidate pairs and print them in decreasing order
'''
print("\nJaccard Similarity")
for c in range(len(candidate_list)):
  print("Doc" + str(j[c][1]+1) +" "+ str(j[c][0]))

end = timeit.default_timer()
print('\nRetrieval: ', end - start)
