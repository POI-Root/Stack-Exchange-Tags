import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from time import clock
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer

# local method
from frequent_stop_words import getStopWords

def clean_content(row) :
    string = row["content"]
    string = "".join(string.split("\n"))
    soup = BeautifulSoup(string, "html.parser")
    return " ".join(re.findall("[a-zA-Z]+", soup.get_text().lower()))
def clean_title(row) :
    string = row["title"]
    return " ".join(re.findall("[a-zA-Z]+", string.lower()))
def words_compound(x) :
	return x.replace(" ", "-")
def get_output_tags(row) :
	
	tags = ""
	vectorizer = CountVectorizer(min_df = 1, stop_words = stop_words, ngram_range = (1, 3))
	try : # empty vocabulary; perhaps the documents only contain stop words
		title_words = set(vectorizer.fit([row["title"]]).get_feature_names())
	except :
		title_words = set()
	try : # empty vocabulary; perhaps the documents only contain stop words
		content_words = set(vectorizer.fit([row["content"]]).get_feature_names())
	except :
		content_words = set()
	both = title_words | content_words
	
	# find words in both
	remove = set()
	count = 0
	tags_dict = dict()
	for word in both :
		if " " in word and len(word.split()) == 3 :
			#remove.add(word)
			word = words_compound(word)
			if word in set(compound_3rd) :
				tags += word + " "
				count += 1
				word = word.split("-")
				remove.add(" ".join([word[0], word[1]]))
				remove.add(" ".join([word[1], word[2]]))
	for word in both :
		if " " in word and len(word.split()) == 2 :
			if word not in remove :
				# learn from 3rd compound
				if word == "speed light" :
					tags += "speed-of-light "
					count += 1
					for w in word.replace("-"," ").split() :
						remove.add(w)
					continue
				if word == "moment inertia" :
					tags += "moment-of-inertia "
					count += 1
					for w in word.replace("-"," ").split() :
						remove.add(w)
					continue
				word = words_compound(word)
				if word in set(compound) :
					tags += word + " "
					count += 1
					for w in word.replace("-"," ").split() :
						remove.add(w)
	for word in both :
		if " " not in word :
			if word not in remove :
				if word in set(single_title) :
					rank = single_title.index(word) + 1
					#if word in set(single_content) :
					#	tags_dict[word] = 1 / rank
					#else :
					tags_dict[word] = (-1) * rank
			#else :
			#	if word in single_content :
			#		tags_dict[word] = 0.4
	if 3 - count > 0 :
		pred_tags = sorted(tags_dict, key=tags_dict.get, reverse=True)[:3-count]
	else :
		pred_tags = []
		return tags[:-1]

	return tags + " ".join(pred_tags)

def get_title_tags(row) :

	tfrequency_dict = defaultdict(int)
	word_count = 0.
	for word in row["title"].split() :
		if word not in stop_words and len(word) > 3:
			tfrequency_dict[word] += 1
			word_count += 1.
	for word in tfrequency_dict:
		tf = tfrequency_dict[word] / word_count
		tfrequency_dict[word] = tf 
	pred_title_tags = sorted(tfrequency_dict, key=tfrequency_dict.get, reverse=True)[:10]
	return " ".join(pred_title_tags)

def get_compound_tags(row) :

	vectorizer = CountVectorizer(min_df = 1, stop_words = stop_words, ngram_range = (3, 3))
	try : # empty vocabulary; perhaps the documents only contain stop words
		title_words = set(vectorizer.fit([row["title"]]).get_feature_names())
	except :
		title_words = set()
	try : # empty vocabulary; perhaps the documents only contain stop words
		content_words = set(vectorizer.fit([row["content"]]).get_feature_names())
	except :
		content_words = set()
	both = title_words & content_words
	# find compound words

	for word in both :
		if " " in word :
			s = word.split()
			if len(s) > 2 :
				#if len(s[0]) > 2 and len(s[1]) > 2 and len(s[2]) > 2:
				tags += words_compound(word) + " "
	return tags[:-1]

df = pd.read_csv("input/small_test.csv")
print("Start data preprocesssing")
df["title"] = df.apply(clean_title, axis = 1)
df["content"] = df.apply(clean_content, axis = 1)
#print(df)
stop_words = set(getStopWords())
f = open("input/verb_stop_words.txt")
verbs = set(f.read().split(","))
f = open("input/color_stop_words.txt")
colors = set(f.read().split(","))
f = open("input/compound_tags_test.txt")
compound = f.read().split("\n")[:-13]
f = open("input/3rd_compound_tags.txt")
compound_3rd = f.read().split("\n")[:4]
f = open("input/single_title_tags.txt")
single_title = f.read().split("\n")[:-14]
f = open("input/single_content_tags.txt")
single_content = f.read().split("\n")
special_stop_words = set(["question", "questions", "problem", "problems", "difference", "correct", "doesn"])
stop_words = stop_words | verbs | special_stop_words #| colors
s = clock()
print("Start predict")
output = df.apply(get_output_tags, axis = 1)
print("time =", clock() - s)
submission = pd.DataFrame({"id" : df["id"], "tags" : output})
submission.to_csv('output_3rd.csv', index=False)
