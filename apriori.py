import sys
import csv
import string
import operator
import random
import re
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
#punctuations
punc = set(string.punctuation)
freq = {} #freq dic
CL = {"yes": 0, "no": 0}
CPDC = {"yes": {}, "no": {}}
CPDL = {}



def regex_split(s):
	rgx = re.compile("([\w][\w']*\w)")
	return rgx.findall(s)

def calcfreq(lowerStr):
	for word in lowerStr:
		if word in freq:
			freq[word]+=1
		else:
			freq[word]=1

def clearpunc(string):
	result=""
	for ch in string:
		if ch not in punc:
			result=result+ch
		# if ch in punc:
		# 	result=result+" "
	return result
def loadcsv(filename):
	f=open(filename, "rb")
	lines = csv.reader(f)
	dataset = list(lines)
	f.close()
	return dataset

if __name__ == '__main__':
	review=[]
	classlabelset=[]
	training="stars_data.csv"
	# testing=sys.argv[2]
	classlabelindex=7
	dataset=loadcsv(training)
	trainsize=len(dataset)
	# NBC model
	for i in range(1,len(dataset)):
		#cleanStr=clearpunc(dataset[i][7]) # hard coded for review column
		classlabel=int(dataset[i][classlabelindex-1])
		cleanStr = dataset[i][7]
		lowerStr=[s.lower() for s in regex_split(cleanStr)]
		review.append(set(lowerStr))  # might cause problem ???????????????????????
		classlabelset.append(classlabel)
		calcfreq(lowerStr)
	#sort words in descending order
	sortedfreq = sorted(freq.items(), key=operator.itemgetter(1),reverse=True)
	selected = sortedfreq[100:2100] # drop the top 100 words
	selectedWords = [item[0] for item in selected]
	# topwords = set(selectedWords) #for speeding
	topWordToIndex = {}
	i = 0
	for word in selectedWords:
		topWordToIndex[word]=i
		i+=1

	# build matrix
	feature_matrix=[[0 for x in range(2500)] for x in range(2000)]
	for i in range(1, len(dataset)):
		for word in r:
			if word in topwords:
				yesmatrix[topWordToIndex[word]][yesindex]+=1
		yesindex+=1
	selectedWords