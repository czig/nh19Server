import pandas as pd
import numpy as np
from os import path
from PIL import Image
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
import itertools
import collections
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
import re
import networkx as nx
import warnings
import string
import unicodedata
from gensim import corpora
import gensim
import pyLDAvis
import pyLDAvis.gensim
import warnings
import argparse
import spacy
from gensim.models.coherencemodel import CoherenceModel 

#initialize sentiment analysis
sid = SentimentIntensityAnalyzer()

#importing stop words
stopwords = set(stopwords.words('english'))

#initiate sql database
engine = create_engine('sqlite:///./combined_surveys.db')
out_engine = create_engine('sqlite:///./survey_sentiment.db')

entry_df = pd.read_sql("""select * from entry_surveys""", engine)
camp_df = pd.read_sql("""select * from exit_surveys""", engine)
exit_df = pd.read_sql("""select * from exit_surveys""",engine)
#Function to do this over and over 
def makedf(df, Comment):
	poop = df[Comment].reset_index()
	poop.columns = ['Index', 'Comments']
	poop['neg'] = 0.0
	poop['neu'] = 0.0
	poop['pos'] = 0.0
	poop['compound'] = 0.0
	poop = poop[(poop.Comments != "")].dropna()
	for index, row in poop.iterrows():
		sentiment = sid.polarity_scores(row['Comments'])
		poop.at[index,'pos'] = sentiment['pos']
		poop.at[index,'neu'] = sentiment['neu']
		poop.at[index,'neg'] = sentiment['neg']
		poop.at[index,'compound'] = sentiment['compound']
	poop.name = Comment
	#print(poop)
	return poop

#Function to make word clouds
def wc(df,pic):
	text = " ".join(comment for comment in df.Comments)
	wc = WordCloud(width = 1000, height = 600, stopwords = stopwords, background_color = "white").generate(text)
	fig = plt.figure(1,figsize = (30,30))
	plt.imshow(wc, interpolation = 'bilinear')
	plt.axis("off")
	plt.title(pic, fontsize = 96)
	fig.savefig(pic,bbox_inches = 'tight')
	plt.close(fig)

#Function to make bigram networks
def bigram_gen(df,pic):
	warnings.filterwarnings('ignore')
	sns.set(font_scale = 1.5)
	sns.set_style("whitegrid")
	text_list = df.Comments.tolist()
	text_np = [comment.translate(str.maketrans('','',string.punctuation)) for comment in text_list]
	text = [comment.lower().split() for comment in text_np]
	text_nsw = [[word for word in comment_words if not word in stopwords] for comment_words in text]

	terms_bigram = [list(bigrams(comments)) for comments in text_nsw]
	bigram = list(itertools.chain(*terms_bigram))
	bigram_counts = collections.Counter(bigram)
	bigram_df = pd.DataFrame(bigram_counts.most_common(50), columns = ['bigram', 'count'])
	bigram_df = bigram_df[bigram_df['count']>1]
	print(bigram_df)
	
	d = bigram_df.set_index('bigram').T.to_dict('records')
	G = nx.Graph()
	for k,v in d[0].items():
		G.add_edge(k[0],k[1],weight = (v*10))
	fig, ax = plt.subplots(figsize = (10,8))
	pos = nx.spring_layout(G,k=1)

	nx.draw_networkx(G,pos,font_size=16,width=3,edge_color='grey',node_color='purple',with_labels=False,ax=ax)
	plt.title(pic, fontsize = 96)
	for key, value in pos.items():
		x,y = value[0]+0.05,value[1]+0.01
		ax.text(x,y,s=key,bbox=dict(facecolor='red',alpha=0.25),horizontalalignment='center',fontsize=13)
	fig.savefig(pic,bbox_inches='tight')
	plt.close(fig)

def wordDist(df):
	text = " ".join(comment for comment in df.Comments)
	words = nltk.word_tokenize(text)
	words = [word for word in words if len(word)>1]
	words = [word for word in words if word not in stopwords]
	words = [word.lower() for word in words]
	print(words)
	fdist = nltk.FreqDist(words)
	for word, frequency in fdist.most_common(100):
		print(u'{}:{}'.format(word, frequency))


def topicModel(df, numtopics):
	arg_parser = argparse.ArgumentParser()
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
	args = arg_parser.parse_args()
	if args.ignore:
		print('Ignoring all warnings...')
		warnings.filterwarnings("ignore")

	parser = spacy.load('en_core_web_sm')
	poop = df[df['Comments'].notnull()].Comments.to_list()
	print('Number of comments:', len(poop))
	def tokenize(messages_list, parser):
		docs_list = []
		for message in messages_list:
			lda_tokens = []
			message = message.replace(chr(8216),"'")
			message = message.replace(chr(8217),"'")
			message = message.replace(chr(8218),",")
			message = message.replace(chr(8220),'"')
			message = message.replace(chr(8221),'"')
			message = message.replace(chr(8242),'`')
			message = message.replace(chr(8245),'`')
			message = unicodedata.normalize('NFKD',message).encode('ascii','ignore').decode('utf-8')
			allowed_pos = ['NOUN','VERB','PROPN']
			possessive_substr = chr(8217)+'s'
			message_tokens = parser(message)
			for token in message_tokens:
				if token.orth_.isspace():
					continue
				elif token.is_punct:
					continue
				elif token.like_url:
					continue
				elif token.like_email:
					continue
				elif token.is_stop:
					continue
				elif token.text.find(possessive_substr) > -1:
					continue
				elif len(token.text) < 2:
					continue
				elif token.pos_ not in allowed_pos:
					continue
				elif token.text in ['member', 'deploy', 'deployed', 'deployment','Guyana']:
					continue
				else:
					lda_tokens.append(token.lemma_)
			docs_list.append(lda_tokens)
		return docs_list
	def compute_coherence_values(dictionary, corpus, texts, limit, start = 2, step = 3):
		coherence_values = []
		model_list = []
		for num_topics in range(start, limit, step):
			model = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word = dictionary, passes = 10, update_every = 1, chunksize = 2, iterations = 1500)
			model_list.append(model)
			coherencemodel = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence = 'u_mass')
			print(coherencemodel.get_coherence())
			coherence_values.append(coherencemodel.get_coherence())
		return model_list, coherence_values

	docs_list = tokenize(poop, parser)
	
	dictionary = corpora.Dictionary(docs_list)
	corpus = [dictionary.doc2bow(doc) for doc in docs_list]

	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = numtopics, id2word = dictionary, passes = 300, update_every = 0, chunksize = 10, iterations = 1500)
	coherencemodel = CoherenceModel(model = ldamodel, texts = docs_list, dictionary = dictionary, coherence = 'u_mass')
	print(coherencemodel.get_coherence())
	model_list, coherence_values = compute_coherence_values(dictionary = dictionary, corpus = corpus, texts = docs_list, start = 2, limit = 32, step = 2)
	limit = 32
	start = 2
	step = 2
	x = range(start, limit, step)
	plt.plot(x, coherence_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Coherence score")
	plt.show()
	topics = ldamodel.print_topics(num_words = 5)
	for topic in topics:
		print(topic)
	vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, R = 10)
	pyLDAvis.save_html(vis, 'lda_vis_{0}_2.html'.format(df.name))

#Creating dataframes
#entrySurveys
homeSupport = makedf(entry_df,'homeSupportComments')
afsouthSupport = makedf(entry_df,'afsouthSupportComments')
adequateTime = makedf(entry_df,'adequateTimeComments')
deployInfo = makedf(entry_df,'deployInfoComments')
readInstructions = makedf(entry_df,'readInstructionsComments')
entry_additional = makedf(entry_df,'additionalComments')
deployAbility = makedf(exit_df, 'deployAbilityComments')
conductingForeign = makedf(exit_df, 'conductingForeignComments')
otherServices = makedf(exit_df, 'otherServicesComments')
partnerNation = makedf(exit_df, 'partnerNationComments')
knowledge = makedf(exit_df, 'knowledgeComments')
utilization = makedf(exit_df, 'utilizationComments')
training = makedf(exit_df, 'trainingComments')
deployedEnv = makedf(exit_df,'deployedEnvComments')
timelyEquipment = makedf(exit_df, 'timelyEquipmentComments')
neededEquipment = makedf(exit_df, 'neededEquipmentComments')
planningRating = makedf(exit_df, 'planningRatingComments')
commNetworks = makedf(exit_df, 'commNetworksComments')
communicate = makedf(exit_df, 'communicateComments')
socialExchanges = makedf(exit_df, 'socialExchangesComments')
professionalExchanges = makedf(exit_df, 'professionalExchangesComments')
livingConditions = makedf(exit_df, 'livingConditionsComments')
healthNeeds = makedf(exit_df, 'healthNeedsComments')
camp_additional = makedf(camp_df, 'additionalComments')
exit_additional = makedf(exit_df, 'additionalComments')
#print(additional)


#Word Cloud Generation
#wc(homeSupport, 'homeSupportWC.png')
#wc(afsouthSupport, 'afsouthSupportWC.png')
#wc(adequateTime, 'adequateTimeWC.png')
#wc(deployInfo,'deployInfoWC.png')
#wc(readInstructions,'readInstructionsWC.png')
#wc(entry_additional,'entry_additionalWC.png')
#wc(exit_additional,'exit_additionalWC.png')
wc(camp_additional, 'camp_additionalWC.png')

#Bigram Network generation
#bigram_gen(homeSupport,'homeSupportBigram.png')
#bigram_gen(afsouthSupport, 'afsouthSupportBigram.png')
#bigram_gen(adequateTime, 'adequateTimeBigram.png')
#bigram_gen(deployInfo, 'deployInfoBigram.png')
#bigram_gen(readInstructions,'readInstructionsBigram.png')
#bigram_gen(entry_additional,'entry_additionalBigram.png')
#bigram_gen(exit_additional, 'exit_additionalBigram.png')
bigram_gen(camp_additional, 'camp_additionalBigram.png')

#Check

#wordDist(exit_additional)
wordDist(camp_additional)

#Topic Modeling
topicModel(camp_additional, numtopics = 3)
print('I did it!')
