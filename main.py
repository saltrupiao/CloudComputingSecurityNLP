# ******** REFERENCE: https://importsem.com/evaluate-sentiment-analysis-in-bulk-with-spacy-and-python/ ************

# Import Statements
import spacy
from spacy import displacy

import pandas as pd

from spacytextblob.spacytextblob import SpacyTextBlob
import en_core_web_trf
from bs4 import BeautifulSoup
import requests

nlp = spacy.load("en_core_web_sm")
nlp = en_core_web_trf.load()


def spacy_init_list_tst():
    # doc = nlp("This is a sentence.")
    # print([(w.text, w.pos_) for w in doc])

    i = 0
    j = 0
    urls = []
    while i == 0:
        print("value for J: ")
        print(j)
        print("Please enter URLs for NLP Processing with Spacy Library: ")
        urls.append(input())
        print("Any other URLs for processing? Enter Y for yes, and N for no: ")
        decision = input()
        if decision == "Y":
            i = 0
        else:
            i = 1
        j += 1
        print("value of J: ")
        print(j)

    print("URLs List: ")
    print(urls)


def visualize_data():
    page_text = "Amazon Web Services (AWS) instance types, including the high-performance Linpack (HPL) benchmark for compute performance. However, the study mostly considers single instances rather than clusters. In particular, while the cost analysis considers the cluster performance of two instance types, this performance appears to be based only on the HPL benchmark, which does not stress the network bandwidth at these sizes, and thus the analysis may not apply to communication intensive applications. The present study attempts to find a more complete look at cluster performance by considering a range of benchmarks with a more varied compute to communication load. It also expands on the previous work by considering a new AWS instance type as well as the Microsoft Azure cloud platform."
    doc = nlp(page_text)
    sentence_spans = list(doc.sents)
    displacy.serve(sentence_spans, style='ent')

    # displacy.render(doc, style='dep', jupyter=True, options = {'distance':100})


def visualize_data2():
    import spacy
    from spacy import displacy
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    i = 0
    j = 0
    urls = []
    while i == 0:
        print("value for J: ")
        print(j)
        print("Please enter URLs for NLP Processing with Spacy Library: ")
        urls.append(input())
        print("Any other URLs for processing? Enter Y for yes, and N for no: ")
        decision = input()
        if decision == "Y":
            i = 0
        else:
            i = 1
        j += 1
        print("value of J: ")
        print(j)

    print("URLs List: ")
    print(urls)

    file_in = pd.read_csv("cc_urls_v1.csv")
    print("printing variable file_in")
    print(file_in)
    ccURLs = file_in["Address"].tolist()
    print("printing ccURLs...")
    print(ccURLs)
    url_sent_score = []
    url_sent_label = []
    total_pos = []
    total_neg = []

    print("Enumerate CCURLs Value: ")
    print(enumerate(ccURLs))

    # for count, x in enumerate(ccURLs):


    url = "https://news.ycombinator.com/item?id=19484167"
    print("Running and scraping for the following URL: ", url)
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
    res = requests.get(url,headers=headers)
    html_page = res.text

    soup = BeautifulSoup(html_page, 'html.parser')
    for script in soup(["script", "style","meta","label","header","footer"]):
        script.decompose()
    page_text = (soup.get_text()).lower()
    page_text = page_text.strip().replace("  ", "")
    page_text = "".join([s for s in page_text.splitlines(True) if s.strip("\r\n")])

    doc = nlp(page_text)
    sentiment = doc._.blob.polarity
    sentiment = round(sentiment, 2)

    if sentiment > 0:
        sent_label = "Positive"
    else:
        sent_label = "Negative"

    url_sent_label.append(sent_label)
    url_sent_score.append(sentiment)

    positive_words = []
    negative_words = []

    for x in doc._.blob.sentiment_assessments.assessments:
        if x[1] > 0:
            positive_words.append(x[0][0])
        elif x[1] < 0:
            negative_words.append(x[0][0])
        else:
            pass

    total_pos.append(', '.join(set(positive_words)))
    total_neg.append(', '.join(set(negative_words)))

    print(file_in)
    print(url_sent_score)
    print(url_sent_label)
    print(total_pos)
    print(total_neg)

    file_in["Sentiment Score"] = pd.Series([url_sent_score])
    file_in["Sentiment Label"] = pd.Series([url_sent_label])
    file_in["Positive Words"] = pd.Series([total_pos])
    file_in["Negative Words"] = pd.Series([total_neg])


    # optional export to CSV
    file_in.to_csv('sentiment_trial8.csv')
    file_in

    # print(doc._.blob.polarity)
    # print(doc._.blob.subjectivity)
    # print(doc._.blob.sentiment_assessments.assessments)
    # sentence_spans = list(doc.sents)
    # displacy.serve(sentence_spans, style='ent')


def visualize_data3():
    import spacy
    from spacy import displacy
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    i = 0
    j = 0
    urls = []
    while i == 0:
        print("value for J: ")
        print(j)
        print("Please enter URLs for NLP Processing with Spacy Library: ")
        urls.append(input())
        print("Any other URLs for processing? Enter Y for yes, and N for no: ")
        decision = input()
        if decision == "Y":
            i = 0
        else:
            i = 1
        j += 1
        print("value of J: ")
        print(j)

    print("URLs List: ")
    print(urls)

    file_in = pd.read_csv("cc_urls_v1.csv")
    print("printing variable file_in")
    print(file_in)
    ccURLs = file_in["Address"].tolist()
    print("printing ccURLs...")
    print(ccURLs)
    url_sent_score = []
    url_sent_label = []
    total_pos = []
    total_neg = []

    print("Enumerate CCURLs Value: ")
    print(enumerate(ccURLs))

    # for count, x in enumerate(ccURLs):


    url = "https://news.ycombinator.com/item?id=19484167"
    print("Running and scraping for the following URL: ", url)
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
    res = requests.get(url,headers=headers)
    html_page = res.text

    soup = BeautifulSoup(html_page, 'html.parser')
    for script in soup(["script", "style","meta","label","header","footer"]):
        script.decompose()
    page_text = (soup.get_text()).lower()
    page_text = page_text.strip().replace("  ", "")
    page_text = "".join([s for s in page_text.splitlines(True) if s.strip("\r\n")])

    doc = nlp(page_text)
    sentiment = doc._.blob.polarity
    sentiment = round(sentiment, 2)

    if sentiment > 0:
        sent_label = "Positive"
    else:
        sent_label = "Negative"

    url_sent_label.append(sent_label)
    url_sent_score.append(sentiment)

    positive_words = []
    negative_words = []

    for x in doc._.blob.sentiment_assessments.assessments:
        if x[1] > 0:
            positive_words.append(x[0][0])
        elif x[1] < 0:
            negative_words.append(x[0][0])
        else:
            pass

    total_pos.append(', '.join(set(positive_words)))
    total_neg.append(', '.join(set(negative_words)))

    print(file_in)
    print(url_sent_score)
    print(url_sent_label)
    print(total_pos)
    print(total_neg)

    file_in["Sentiment Score"] = pd.Series([url_sent_score])
    file_in["Sentiment Label"] = pd.Series([url_sent_label])
    file_in["Positive Words"] = pd.Series([total_pos])
    file_in["Negative Words"] = pd.Series([total_neg])


    # optional export to CSV
    file_in.to_csv('sentiment_trial8.csv')
    file_in

    # print(doc._.blob.polarity)
    # print(doc._.blob.subjectivity)
    # print(doc._.blob.sentiment_assessments.assessments)
    # sentence_spans = list(doc.sents)
    # displacy.serve(sentence_spans, style='ent')


def main():
    print("Running Init Function\n")
    spacy_init()


main()

