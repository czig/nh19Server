import spacy
from gensim.models.phrases import Phrases, Phraser
import unicodedata
import re

class Tokenizer(object):
    """Class for tokenizing documents.

    The __init__ method sets allowed_pos to all possible spacy parts of speech (POS) if an
    empty list is supplied for allowed_pos. Additionally, the stop words are added to the 
    parser in __init__.

    Args:
        model (str, optional): spacy model for parser. Defaults to 'en_core_web_sm'
        stop_words (list of str, optional): words to remove when tokenizing. Defaults to
            empty list.
        case_sensitive (bool, optional): if False, make lowercase version of all stop_words stop 
            words, if True, exact stop_word case is used. Defaults to True.
        stop_lemmas (list of str, optional): list of lemmas to remove when tokenizing. All items
            in this list get converted to lowercase. Defaults to empty list.
        remove_unicode (bool, optional): if True, converts unicode to closest ascii representation.
            If False, keeps unicode. Defaults to False.
        allowed_pos (list of str, optional): spacy parts of speech (POS) to allow when tokenizing. Defaults
            to empty list, which results in all POS being allowed.
        remove_pos (list of str, optional): spacy parts of speech (POS) to remove when tokenizing. Inverse
            of allowed_pos for usability. Defaults to empty list.
        lemma_token (bool, optional): if True, uses lemma for token, if False, uses original text for token.
            Defaults to True.
        lower_token (bool, optional): if True, takes the lowercase of the token, if False, keeps original case
            of token. Defaults to False.
        bigrams (bool, optional): if True, appends common bigrams to each message. if false, does not append 
            bigrams. Defaults to False.
        bigram_min_count (int, optional): minimum amount of times bigram must appear in all documents before 
            it is appended to message (assuming bigrams is True). Defaults to 20.

    Attributes:
        model (str): spacy model for parser. Defaults to 'en_core_web_sm'
        stop_words (list of str): words to remove when tokenizing. Defaults to
            empty list.
        case_sensitive (bool): if False, make lowercase version of all stop_words stop 
            words, if True, exact stop_word case is used. Defaults to True.
        stop_lemmas (list of str): list of lemmas to remove when tokenizing. All items in this list
            get converted to lowercase. Defaults to empty list.
        remove_unicode (bool): if True, converts unicode to closest ascii representation.
            If False, keeps unicode. Defaults to False.
        allowed_pos (list of str): spacy parts of speech (POS) to allow when tokenizing. Defaults
            to empty list, which results in all POS being allowed.
        remove_pos (list of str): spacy parts of speech (POS) to remove when tokenizing. Inverse
            of allowed_pos for usability. Defaults to empty list.
        lemma_token (bool): if True, uses lemma for token, if False, uses original text for token.
            Defaults to True.
        lower_token (bool): if True, takes the lowercase of the token, if False, keeps original case
            of token. Defaults to False.
        bigrams (bool): if True, appends common bigrams to each message. if false, does not append 
            bigrams. Defaults to False.
        bigram_min_count (int): minimum amount of times bigram must appear in all documents before 
            it is appended to message (assuming bigrams is True). Defaults to 20.

    """
    def __init__(self, model='en_core_web_sm', stop_words=[], case_sensitive=True, stop_lemmas=[], remove_unicode=False, allowed_pos=[], remove_pos=[], lemma_token=True, lower_token=False, bigrams=False, bigram_min_count=20):
        #initialize parameters
        self.parser = spacy.load(model)
        self.stop_words = stop_words
        self.case_sensitive = case_sensitive
        self.stop_lemmas = [lemma.lower() for lemma in stop_lemmas]
        self.remove_unicode = remove_unicode
        if len(allowed_pos) == 0:
            self.allowed_pos = [item for item in spacy.parts_of_speech.univ_pos_t.__members__]
        else:
            self.allowed_pos = allowed_pos
        self.remove_pos = remove_pos
        self.lemma_token = lemma_token
        self.lower_token = lower_token
        #establish stop words
        for word in stop_words:
            self.parser.vocab[word].is_stop = True
            if self.case_sensitive == False:
                self.parser.vocab[word.lower()].is_stop = True
        self.parser.vocab["not"].is_stop = False
        self.bigrams = bigrams
        self.bigram_min_count = bigram_min_count

    #tokenize and clean posts
    def tokenize(self, messages_list, return_docs=True):
        """Tokenize a list of messages.

        Note:
            This function can take awhile to execute, especially when the size of messages_list
            is large (>1000s). 

        Args:
            messages_list (list of list of str): list of lists where each list is a document/message 
                to be tokenized
            return_docs (bool, optional): if True, returns list of lists of tokens, where each 
                sublist represents a document/message. if False, returns a list of tokens. Defaults 
                to True.

        Returns:
            list of lists: if return_docs is True
            list: if return_docs is False

        """
        docs_list = []
        for message in messages_list:
            clean_tokens = []
            #convert unicode punctuation to regular ascii punctuation 
            message = message.replace(chr(8216),"'")
            message = message.replace(chr(8217),"'")
            message = message.replace(chr(8218),",")
            message = message.replace(chr(8220),'"')
            message = message.replace(chr(8221),'"')
            message = message.replace(chr(8242),'`')
            message = message.replace(chr(8245),'`')
            #convert unicode to closest ascii
            if self.remove_unicode == True:
                message = unicodedata.normalize('NFKD',message).encode('ascii','ignore').decode('utf-8')
            #unicode for 's (right apostrophie followed by s)
            possessive_substr = chr(8217) + 's'
            message_tokens = self.parser(message)
            #iterate through all tokens in each post
            for token in message_tokens:
                #remove space
                if token.orth_.isspace():
                    continue
                #remove punctuation
                elif token.is_punct:
                    continue
                #remove urls
                elif token.like_url:
                    continue
                #remove emails
                elif token.like_email:
                    continue
                #remove stop words
                elif token.is_stop:
                    continue
                #remove 's
                elif token.text.find(possessive_substr) > -1:
                    continue
                #remove single letters
                elif len(token.text) < 2:
                    continue
                #only keep allowed parts of speech
                elif token.pos_ not in self.allowed_pos:
                    continue
                #remove unwanted parts of speech 
                elif token.pos_ in self.remove_pos:
                    continue
                #remove certain lemmas
                elif token.lemma_.lower() in self.stop_lemmas:
                    continue
                #remove phone numbers
                elif re.match(r'\+\d+',token.text):
                    continue
                #remove any single letter followed by a period
                elif re.match(r'[a-zA-Z]\.',token.text):
                    continue
                else:
                    #use lemma to get root word 
                    if self.lemma_token: 
                        #use lowercase lemma
                        if self.lower_token:
                            clean_tokens.append(token.lemma_.lower())
                        #use same case with lemma
                        else:
                            clean_tokens.append(token.lemma_)
                    #do not use lemma, use original word
                    else:
                        #use lowercase original text
                        if self.lower_token:
                            clean_tokens.append(token.text.lower())
                        #use same case with original word
                        else:
                            clean_tokens.append(token.text)
            
            docs_list.append(clean_tokens)

        if self.bigrams:
            #if bigram occurs more than bigram_min_count times in all documents, append bigram to document (message) tokens list
            bigram = Phraser(Phrases(docs_list, min_count=self.bigram_min_count))
            for idx in range(len(docs_list)):
                for token in bigram[docs_list[idx]]:
                    if '_' in token:
                        docs_list[idx].append(token)
        
        if return_docs:
            # return list of lists, where each sublist is the tokens for a document (message)
            return docs_list 
        else:
            # return list of tokens (flattened doc_list)
            return [token for doc in docs_list for token in doc]

