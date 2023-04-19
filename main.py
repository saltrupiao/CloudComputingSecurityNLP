# ******** REFERENCE: Spacy/NLP Code referenced from: https://importsem.com/evaluate-sentiment-analysis-in-bulk-with-spacy-and-python/ ************

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
    page_text = "Amazon Web Services (AWS) instance types, including the high-performance Linpack (HPL) benchmark."
    doc = nlp(page_text)
    sentence_spans = list(doc.sents)
    displacy.serve(sentence_spans, style='ent')

    # displacy.render(doc, style='dep', jupyter=True, options = {'distance':100})


def go1():
    import spacy
    from spacy import displacy
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    i = 0
    j = 0
    urls = []
    while i == 0:
        # print("value for J: ")
        # print(j)
        print("Please enter URLs for NLP Processing with Spacy Library: ")
        urls.append(input())
        print("Any other URLs for processing? Enter Y for yes, and N for no: ")
        decision = input()
        if decision == "Y":
            i = 0
        else:
            i = 1
        j += 1
        # print("value of J: ")
        # print(j)

    print("URLs List: ")
    print(urls)

    for u in urls:
        url_sent_score = []
        url_sent_label = []
        total_pos = []
        total_neg = []
        # url = "https://news.ycombinator.com/item?id=19484167"
        print("Running and scraping for the following URL: ", u)
        headers = {
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
        res = requests.get(u, headers=headers)
        html_page = res.text

        soup = BeautifulSoup(html_page, 'html.parser')
        for script in soup(["script", "style", "meta", "label", "header", "footer"]):
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

        print("\n\n *************** SENTIMENT SCORE REPORT FOR URL " + u + " ***************")
        print("\n\nSentiment Score: ")
        print(url_sent_score)
        print("\nSentiment Label: ")
        print(url_sent_label)
        print("\nPositive Words: ")
        print(total_pos)
        print("\nNegative Words: ")
        print(total_neg)
        print("\n\n *************** END OF SENTIMENT SCORE REPORT FOR URL " + u + " ***************")


def main():
    print("Running Init Function\n")
    go1()


main()

