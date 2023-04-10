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


def spacy_init():
    doc = nlp("This is a sentence.")
    print([(w.text, w.pos_) for w in doc])


def visualize_data():
    doc = nlp(u'Tesla to build solar electric startup in gujrat for $70 million')
    displacy.render(doc, style='dep', jupyter=True, options = {'distance':100})


def visualize_data2():
    import spacy
    from spacy import displacy
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    file_in = pd.read_csv("cc_urls_v1.csv")
    ccURLs = file_in["Address"].tolist()
    url_sent_score = []
    url_sent_label = []
    total_pos = []
    total_neg = []

    for count, x in enumerate(ccURLs):
        url = x
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

        file_in["Sentiment Score"] = pd.Series([url_sent_score])
        file_in["Sentiment Label"] = pd.Series([url_sent_label])
        file_in["Positive Words"] = pd.Series([total_pos])
        file_in["Negative Words"] = pd.Series([total_neg])

    # optional export to CSV
    file_in.to_csv("sentiment_trial4.csv")
    file_in

    # print(doc._.blob.polarity)
    # print(doc._.blob.subjectivity)
    # print(doc._.blob.sentiment_assessments.assessments)
    # sentence_spans = list(doc.sents)
    # displacy.serve(sentence_spans, style='ent')


def main():
    print("Running Init Function\n")
    visualize_data2()


main()

