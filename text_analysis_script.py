import nltk
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import glob
import os
from nltk.tokenize import word_tokenize

#Common Method For Reading File
def read_file_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as fp:
            content = fp.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as fp:
            content = fp.read()
    return content

def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as fp:
            content = fp.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as fp:
            content = fp.readlines()
    return content


#Reading excel
data = pd.read_excel('inputxl/Input.xlsx')
#Creating Data Frame For Storing The Scores
df=pd.DataFrame(data)

#Scrapping the website data and creating text files of text data in data folder
def fetchAndSaveToFile(url,path):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    article=soup.find(class_="td-post-content")
    title = soup.find('title')
    if article is not None:
        f = open(path, "w",encoding="utf-8")
        f.write(title.get_text()+article.get_text())

for i in range(0,100):
    url=data['URL'][i]
    filename=data['URL_ID'][i]
    directory = "data"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path=f"{directory}/{filename}"+".txt"
    fetchAndSaveToFile(url,path)

#Creating the list of stopwords for the cleaning of text files
listOf=[]
for file in glob.glob("stopword/*.txt"):
    mulList=read_file(file)
    if listOf is None:
        listOf=mulList
    else:
        listOf.extend(mulList)
newList=[]
for words in listOf:
        neword=words.replace('\n','')
        if len(neword)>40:
            split_result = neword.split("|")
            res = split_result[0]
            neword=res
        newList.extend(neword.split(" | ",1))


#Cleaning the text files by removing stop words from data
for files in glob.glob("data/*.txt"):
    example_sent = read_file_data(files)
    word_tokens = word_tokenize(example_sent,language="english")
    filtered_sentence = [w for w in word_tokens if not w.lower() in newList]
    filename = os.path.basename(files)
    directory = "cleaneddata"
    if not os.path.exists(directory):
        os.makedirs(directory)
    f=open(f"{directory}/{filename}","w",encoding="utf-8")
    lower_filter = [filterS.lower() for filterS in filtered_sentence if filterS.isalpha()]
    unique_filter_sentence = list(set(lower_filter))
    f.write(" ".join(unique_filter_sentence))

#Cleaning Master Dictionary
positive_words = []
negative_words = []
for files in glob.glob("MasterDictionary/*.txt"):
    with open(files, "r") as f:
        masterList = f.read()
        word_tokens = word_tokenize(masterList, language="english")
        filtered_sentence = [w for w in word_tokens if not w.lower() in newList]
        filename = os.path.basename(files)
        if filename.startswith("positive"):
            positive_words=filtered_sentence
        else:
            negative_words=filtered_sentence
#Checking the data words in positive and negative dictionary and calculating scores
for dataFiles in glob.glob("cleaneddata/*.txt"):
    positiveScore = 0
    negativeScore = 0
    polarityScore = 0
    subjectiveScore = 0
    text = read_file_data(dataFiles)
    word_tokens = word_tokenize(text, language="english")
    for word in word_tokens:
        if word.lower() in positive_words:
            positiveScore += 1
        elif word.lower() in negative_words:
            negativeScore += 1
    polarityScore=(positiveScore-negativeScore)/ ((positiveScore + negativeScore) + 0.000001)
    subjectiveScore=(positiveScore+negativeScore)/ (len(word_tokens) + 0.000001)
    basename = os.path.splitext(os.path.basename(dataFiles))[0]
    df.loc[df['URL_ID'] == basename, "POSITIVE SCORE"] = positiveScore
    df.loc[df['URL_ID'] == basename, "NEGATIVE SCORE"] = negativeScore
    df.loc[df['URL_ID'] == basename, "POLARITY SCORE"] = polarityScore
    df.loc[df['URL_ID'] == basename, "SUBJECTIVITY SCORE"] = subjectiveScore

#Analysis of readability
def count_complex_words(text):
    def count_syllables(word):
        return len(re.findall(r'[aeiouAEIOU]+', word))
    complex_word_count = sum(1 for word in text.split() if count_syllables(word) > 2)
    return complex_word_count

for dataFiles in glob.glob("data/*.txt"):
    text = read_file_data(dataFiles)
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(words)
    complex_word_count = count_complex_words(text)
    avgSenLen = num_words/num_sentences
    perComplexWords = complex_word_count/num_words
    fogIndex = 0.4 * (avgSenLen + perComplexWords)
    avgNumWordsPerSen = num_words/num_sentences
    basename = os.path.splitext(os.path.basename(dataFiles))[0]
    df.loc[df['URL_ID'] == basename, "AVG SENTENCE LENGTH"] = avgSenLen
    df.loc[df['URL_ID'] == basename, "PERCENTAGE OF COMPLEX WORDS"] = perComplexWords
    df.loc[df['URL_ID'] == basename, "FOG INDEX"] = fogIndex
    df.loc[df['URL_ID'] == basename, "AVG NUMBER OF WORDS PER SENTENCE"] = avgNumWordsPerSen
    df.loc[df['URL_ID'] == basename, "Complex Word Count"] = complex_word_count

#Word Count of Cleaned Words
for dataFiles in glob.glob("cleaneddata/*.txt"):
    text = read_file_data(dataFiles)
    words = nltk.word_tokenize(text)
    num_words = len(words)
    basename = os.path.splitext(os.path.basename(dataFiles))[0]
    df.loc[df['URL_ID'] == basename, "WORD COUNT"] = num_words

#Counting Syllable Per Word
def count_syllables(word):
    exceptions = ["es", "ed"]
    syllable_count = 0
    prev_vowel = False
    for char in word.lower():
        if char in "aeiou":
            if not prev_vowel:
                syllable_count += 1
            prev_vowel = True
        else:
            prev_vowel = False
    for exception in exceptions:
        if word.endswith(exception):
            syllable_count -= 1
    return max(syllable_count, 1)


def count_syllables_per_word(text):
    words = re.findall(r'\b\w+\b', text)
    syllable_count_word =0
    for word in words:
        syllable_count_word += count_syllables(word)
    return syllable_count_word


for dataFiles in glob.glob("cleaneddata/*.txt"):
    text = read_file_data(dataFiles)
    syllable_count_per_word = count_syllables_per_word(text)
    basename = os.path.splitext(os.path.basename(dataFiles))[0]
    df.loc[df['URL_ID'] == basename, "SYLLABLE PER WORD"] = syllable_count_per_word

#Counting Personal Pronoun
for dataFiles in glob.glob("data/*.txt"):
    text = read_file_data(dataFiles)
    pattern = r'\b(?:I|we|my|ours|us)\b'
    allMatches = re.findall(pattern, text, flags=re.IGNORECASE)
    matches = [match for match in allMatches if match != 'US']
    basename = os.path.splitext(os.path.basename(dataFiles))[0]
    df.loc[df['URL_ID'] == basename, "PERSONAL PRONOUNS"] = len(matches)

#Counting Average Word Length
for dataFiles in glob.glob("cleaneddata/*.txt"):
    text = read_file_data(dataFiles)
    words = nltk.word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    avgWordLen=total_characters/len(words)
    basename = os.path.splitext(os.path.basename(dataFiles))[0]
    df.loc[df['URL_ID'] == basename, "AVG WORD LENGTH"] = avgWordLen

#Creating Output Excel
dir = "output_data_structure"
if not os.path.exists(dir):
    os.makedirs(dir)
output_path = os.path.join(dir, "output_data.xlsx")
df.to_excel(output_path, index=False)