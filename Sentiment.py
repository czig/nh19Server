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

#input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
args = arg_parser.parse_args()
if args.ignore:
    print('Ignoring all warnings...')
    warnings.filterwarnings("ignore")

#initialize sentiment analysis
sid = SentimentIntensityAnalyzer()

#importing stop words
stopwords = set(stopwords.words('english'))

#initiate sql database
engine = create_engine('sqlite:///./combined_surveys.db')
out_engine = create_engine('sqlite:///./survey_sentiment.db')

entry_df = pd.read_sql("""select * from entry_surveys""", engine)
camp_df = pd.read_sql("""select * from camp_surveys""", engine)
exit_df = pd.read_sql("""select * from exit_surveys""",engine)

#Function to do this over and over 
def makedf(df, Comment):
    comment_df = df[Comment].reset_index()
    comment_df.columns = ['Index', 'Comments']
    comment_df['neg'] = 0.0
    comment_df['neu'] = 0.0
    comment_df['pos'] = 0.0
    comment_df['compound'] = 0.0
    comment_df = comment_df[(comment_df.Comments != "")].dropna()
    for index, row in comment_df.iterrows():
        sentiment = sid.polarity_scores(row['Comments'])
        comment_df.at[index,'pos'] = sentiment['pos']
        comment_df.at[index,'neu'] = sentiment['neu']
        comment_df.at[index,'neg'] = sentiment['neg']
        comment_df.at[index,'compound'] = sentiment['compound']
    comment_df.name = Comment
    #print(comment_df)
    return comment_df

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

