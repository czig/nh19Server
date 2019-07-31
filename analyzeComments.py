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
import argparse
from tokenizer import *
import statistics

#seaborn settings
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

#input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
arg_parser.add_argument("--num_bigrams", help="Number of bigrams to show in barchart and graph. Default is 20.", type=int, default=20)
args = arg_parser.parse_args()
if args.ignore:
    print('Ignoring all warnings...')
    warnings.filterwarnings("ignore")

sid = SentimentIntensityAnalyzer()

#define stopwords
comment_stopwords = ["Air","Force","New","Horizons","Horizon"]

#define and instantiate tokenziers
comment_tokenizer = Tokenizer(stop_words=comment_stopwords, remove_pos=['PRON'], lemma_token=False, lower_token=True)

#initiate sql database
engine = create_engine('sqlite:///./combined_surveys.db')
out_engine = create_engine('sqlite:///./survey_sentiment.db')

#get all data
entry_df = pd.read_sql("""select * from entry_surveys""", engine)
camp_df = pd.read_sql("""select * from camp_surveys""", engine)
exit_df = pd.read_sql("""select * from exit_surveys""",engine)

#get list of columns that are comments
entry_comment_cols = [column for column in list(entry_df.columns) if 'Comments' in column]
camp_comment_cols = [column for column in list(camp_df.columns) if 'Comments' in column]
exit_comment_cols = [column for column in list(exit_df.columns) if 'Comments' in column]
exit_comment_cols.remove('deployedEnvComments')

#select comment columns
entry_comments = entry_df[entry_comment_cols]
camp_comments = camp_df[camp_comment_cols]
exit_comments = exit_df[exit_comment_cols]

#add camp comments to exit comments
exit_comments = pd.concat([exit_comments,camp_comments])

#show size of dataframes 
print('Shape of entry surveys: ',entry_comments.shape)
print('Shape of exit surveys: ',exit_comments.shape)

#Function to make a dataframe for each comment and add sentiment scores 
def make_df(df, column_name):
    comment_df = df[column_name].reset_index()
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
    comment_df.name = column_name
    #print(comment_df)
    return comment_df


def plot_WordFreq(tokens,name,survey):
    word_freq = dict(collections.Counter(tokens).most_common(20))
    fig = plt.figure(figsize=(12.8,9.6), dpi=200)
    plt.bar(word_freq.keys(), word_freq.values())
    plt.xticks(rotation=40, ha='right', rotation_mode='anchor')
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel("Frequency")
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.title(survey + ': ' + name + ' Word Frequency', fontsize=20)
    filename = './wordfreq/' + survey + '_' + name + '_wordfreq.png'
    plt.savefig(filename)
    plt.close(fig)

#Function to make word clouds
def make_wc(tokens,name,survey):
    text = " ".join(tokens)
    wc = WordCloud(width = 1000, height = 800, background_color = "white").generate(text)
    fig = plt.figure(figsize=(18,14))
    plt.imshow(wc, interpolation = 'bilinear')
    plt.axis("off")
    plt.title(survey + ': ' + name, fontsize = 20)
    filename = './wordclouds/' + survey + '_' + name + '_wc.png'
    plt.savefig(filename,bbox_inches = 'tight')
    plt.close(fig)

#function for plotting bigrams
def gen_bigrams(tokens,name,survey):
    #generate top N bigrams and put in easy-to-use form
    bigrams = list(nltk.bigrams(tokens))
    bigram_count = collections.Counter(bigrams).most_common(args.num_bigrams)
    bigram_dict = {item[0]: item[1] for item in bigram_count if item[1] > 1}
    #create bar plot of bigrams
    bar_fig = plt.figure(figsize=(12.8,9.6), dpi=200)
    keys = [bigram[0] + '_' + bigram[1] for bigram in bigram_dict]
    plt.bar(keys, list(bigram_dict.values()))
    plt.xticks(rotation=40, ha='right', rotation_mode='anchor')
    plt.subplots_adjust(bottom=0.3)
    plt.xlabel("Bigram")
    plt.ylabel("Frequency")
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)
    bar_filename = './bigrams/' + survey + '_' + name + '_bigramBar.png'
    plt.title(survey + ': ' + name + ' Top {0} Bigrams'.format(args.num_bigrams), fontsize=20)
    plt.savefig(bar_filename)
    plt.close(bar_fig)
    #create bigram graph
    G = nx.Graph()
    node_dict = {}
    # add all edges (bigrams ) to graph
    for k,v in bigram_dict.items():
        G.add_edge(k[0],k[1], weight=(v*5))
        #populate word frequency from bigrams alone
        if k[0] in node_dict.keys():
            node_dict[k[0]] += v
        else:
            node_dict[k[0]] = v
        if k[1] in node_dict.keys():
            node_dict[k[1]] += v
        else:
            node_dict[k[1]] = v

    fig,ax = plt.subplots(figsize=(18,14), dpi=200)
    pos = nx.spring_layout(G,k=1)
    #draw graph
    nx.draw_networkx(G,pos,font_size=16,width=3,edge_color='grey',node_size=200,node_color=list(node_dict.values()),cmap=plt.cm.plasma,with_labels=False,ax=ax)
    plt.title(survey + ': ' + name + ' Bigram Diagram')
    #turn off grid lines and both axes
    ax.axis('off')
    #label graph
    for key, value in pos.items():
        x = value[0]
        y = value[1]+0.05
        ax.text(x,y,s=key,bbox=dict(facecolor='grey',edgecolor='black',alpha=0.1),horizontalalignment='center',fontsize=13)
    #create colorbar as legend for colormap
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    sm._A = list(node_dict.values())
    cbar = plt.colorbar(sm)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Word Frequency', rotation=270)
    graph_filename = './bigrams/' + survey + '_' + name + '_bigramGraph.png'
    plt.savefig(graph_filename)
    plt.close(fig)
    

###Analyze entry surveys
all_entry_df = {}
entry_mean_sentiments = {}
entry_std_sentiments = {}
for column_name in entry_comment_cols:
    #make df for each column with sentiment scores
    tmp_df = make_df(entry_comments,column_name)
    all_entry_df[column_name] = tmp_df
    #make list of comments for column
    comments = tmp_df.Comments.to_list()
    if len(comments) == 0:
        continue
    else:
        #tokenzie and store in dict
        print('Tokenizing {0} comments on {1} from Entry survey'.format(len(comments),column_name))
        tokens = comment_tokenizer.tokenize(comments, return_docs=False) 
        #plot word frequencies and bigrams
        plot_WordFreq(tokens, column_name, 'Entry')
        make_wc(tokens, column_name, 'Entry')
        gen_bigrams(tokens, column_name, 'Entry')
        #pull off all sentiment scores
        sentiments = tmp_df['compound'].to_list()
        mean_sentiment = statistics.mean(sentiments)
        std_sentiment = statistics.stdev(sentiments)
        entry_mean_sentiments[column_name] = mean_sentiment
        entry_std_sentiments[column_name] = std_sentiment

#plot mean and std of sentiment scores for entry surveys
fig_mean = plt.figure()
plt.bar(entry_mean_sentiments.keys(), entry_mean_sentiments.values())
plt.xticks(rotation=40, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Questions")
plt.ylabel("Average Sentiment")
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
plt.title('Entry: Average Sentiment', fontsize=20)

fig_std = plt.figure()
plt.bar(entry_std_sentiments.keys(), entry_std_sentiments.values())
plt.xticks(rotation=40, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Questions")
plt.ylabel("Sentiment St. Dev.")
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
plt.title('Entry: Sentiment St. Dev.', fontsize=20)

##Analyze exit surveys
all_exit_df = {}
exit_mean_sentiments = {}
exit_std_sentiments = {}
for column_name in exit_comment_cols:
    #make df for each column with sentiment scores
    tmp_df = make_df(exit_comments,column_name)
    all_exit_df[column_name] = tmp_df
    #make list of comments for column
    comments = tmp_df.Comments.to_list()
    if len(comments) == 0:
        continue
    else:
        #tokenize and store in dict
        print('Tokenizing {0} comments on {1} from Exit survey'.format(len(comments),column_name))
        tokens = comment_tokenizer.tokenize(comments, return_docs=False)
        #plot word frequencies and bigrams
        plot_WordFreq(tokens, column_name, 'Exit')
        make_wc(tokens, column_name, 'Exit')
        gen_bigrams(tokens, column_name, 'Exit')
        #pull off all sentiment scores
        sentiments = tmp_df['compound'].to_list()
        mean_sentiment = statistics.mean(sentiments)
        std_sentiment = statistics.stdev(sentiments)
        exit_mean_sentiments[column_name] = mean_sentiment
        exit_std_sentiments[column_name] = std_sentiment

#plot mean and std of sentiment scores for exit surveys
fig_mean = plt.figure()
plt.bar(exit_mean_sentiments.keys(), exit_mean_sentiments.values())
plt.xticks(rotation=40, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Questions")
plt.ylabel("Average Sentiment")
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
plt.title('Exit: Average Sentiment', fontsize=20)

fig_std = plt.figure()
plt.bar(exit_std_sentiments.keys(), exit_std_sentiments.values())
plt.xticks(rotation=40, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Questions")
plt.ylabel("Sentiment St. Dev.")
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
plt.title('Exit: Sentiment St. Dev.', fontsize=20)

#finally, show plots (should only see sentiment plots)
plt.show()
