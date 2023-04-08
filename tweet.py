import pandas as pd
import nltk
from tashaphyne.stemming import ArabicLightStemmer
import warnings as wr
import regex as re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import emoji
wr.filterwarnings("ignore")


def Cleaning(copy_data):
    copy_data['Prediction']=copy_data['Prediction'].replace({'non spam':0,'spam':1})
    copy_data.drop(copy_data.columns[copy_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) #remove unnamed columns . 
    copy_data['sentenses_len'] = 0
    copy_data['puretext'] = 0 
    copy_data['#english_words'] = 0 
    copy_data['#hashtags'] = 0 
    copy_data['#mentioning'] = 0 
    copy_data['#hyperlinks'] = 0 
    copy_data['#numbers'] = 0
    copy_data['#emojis']=0
    
    return copy_data

def tokenization(x,idx): # take data copy
        tokens = nltk.word_tokenize(x['TweetText'][idx])
        return tokens
def word_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    return tokens
def segmentation(x,idx): # take data copy 
    s_tokens = nltk.data.load('tokenizers/punkt/english.pickle')
    sentens = s_tokens.tokenize(x['TweetText'][idx])
    x['sentenses_len'][i] = len(sentens) #save length of sentenses in the csv "data set "  file  
    # print(len(sentens))    
def drop_stop_words(x): #take data copy
    arb_stop_words = set(nltk.corpus.stopwords.words("arabic"))
    tokensOfpureTextWithoutstop=[token for token in x if token not in arb_stop_words]
    return tokensOfpureTextWithoutstop
def stemming_Light(x):
    ArListem = ArabicLightStemmer()
    stemming_Light =[ArListem.light_stem(token) for token in x]
    return stemming_Light
def stemming(x):
    st = nltk.ISRIStemmer()
    stemming_root =[st.stem(token) for token in x]
    return stemming_root

def removing_mentioning(text):
    return re.sub(r"@[a-zA-Z0-9]+",'',text)
def removing_hashtags(text):
    return re.sub(r"#[a-zA-Z0-9أ-ى]+",'',text)
def remove_newlines_tabs(text):
    return ' '.join(text.replace('\n', ' ').replace('\t',' ').split())
def remove_numbers(text):
    return re.sub('\d+', '', text)
def remove_links(text):
    return re.sub(r'https?:\/\/.*[\r\n]*', '', text)
def remove_emojis(text):
    return emoji.replace_emoji(text, '').replace('☻', ' ')
def remove_english(text):
    return re.sub('[A-Za-z]+', '', text)

def spliting (data_,target,test_size=0.33):
    X_train, X_test, y_train, y_test = train_test_split(data_, target, test_size=test_size, random_state=44, shuffle =True)
    return X_train, X_test, y_train, y_test


data=pd.read_excel('data/dataset_sample_200.xlsx') 
data = Cleaning(data)

#Pre processing
for i in range(0,data.shape[0]):
    tokens = tokenization(data , i )
    data['#english_words'][i] = len(re.findall(r'[A-Za-z]+', data['TweetText'][i]))
    data['#hyperlinks'][i] = len(re.findall(r'https?:\/\/.*[\r\n]*', data['TweetText'][i]))
    data['#numbers'][i] = len(re.findall('\d+', data['TweetText'][i]))
    data['#hashtags'][i] = len(re.findall(r"#[a-zA-Z0-9أ-ى]+", data['TweetText'][i]))
    data['#mentioning'][i] = len(re.findall(r"@[a-zA-Z0-9]+", data['TweetText'][i]))
    data['#emojis'][i]=emoji.emoji_count(data['TweetText'][i])
    segmentation(data,i)
    pure_tokens = drop_stop_words(tokens)
    pure_tokens = stemming(pure_tokens)
    data['puretext'][i] = ' '.join(pure_tokens)

data.drop('TweetText',axis=1,inplace=True)

data['puretext'] = data['puretext'].apply(removing_mentioning)
data['puretext'] = data['puretext'].apply(removing_hashtags)
data['puretext'] = data['puretext'].apply(remove_numbers)
data['puretext'] = data['puretext'].apply(remove_links)
data['puretext'] = data['puretext'].apply(remove_emojis)
data['puretext'] = data['puretext'].apply(remove_english)

#Classification


counting_vec =CountVectorizer(binary=True)
features = counting_vec.fit_transform(data['puretext']).astype('int8')
counting =pd.DataFrame(features.toarray(), columns= counting_vec.vocabulary_.keys())
X_train, X_test, y_train, y_test=spliting (counting,data['Prediction'],test_size=0.4)
SVMclf = SVC()
SVMclf.fit(X_train, y_train)
y_pred=SVMclf.predict(X_test)





user_tweet = input("Enter a tweet: ")
tweet='افضل اّشتراك iٌpَtًv🤩 اشترًاكٍ الحين'


tokens = word_tokenizer(user_tweet)
pure_tokens = drop_stop_words(tokens)
pure_tokens=stemming(pure_tokens)
pure_text= ' '.join(pure_tokens)
pure_text= removing_mentioning(pure_text)
pure_text= removing_hashtags(pure_text)
pure_text= remove_numbers(pure_text)
pure_text= remove_links(pure_text)
pure_text= remove_emojis(pure_text)
pure_text= remove_english(pure_text)
tweet_features = counting_vec.transform([pure_text]).astype('int8')
feature_array = tweet_features.toarray()
pred=SVMclf.predict(feature_array)

if pred==1:
    print(f"Tweet is Spam")
else:
    print(f"Tweet is Non-spam")
