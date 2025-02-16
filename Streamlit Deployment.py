# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import streamlit as st
    import os
    import numpy as np
    import pandas as pd
    import pickle
    import warnings

    warnings.filterwarnings('ignore')
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from scipy.sparse import hstack


    # Necessary functions for the pipeline
    # Referenced from: https://stackoverflow.com/a/47091490/4084039
    # This will replace words like "won't" with "will not" and so on

    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)

        return phrase


    stop_words = set(stopwords.words('English'))  # Loading in the stop words

    # Removing words like no, nor and not from the stopwords dict

    stop_words.remove('no')
    stop_words.remove('not')
    stop_words.remove('nor')


    # To get rid of emojis
    # Refernce: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

    def deEmojify(text):
        regrex_pattern = re.compile(pattern="["
                                            u"\U0001F600-\U0001F64F"  # emoticons
                                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                            "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', text)


    # importing necessary pickle files for feature extraction and modelling
    import pickle

    # loading tfidf_title vectorizer
    file = open('title_tfidf.pickle', 'rb')
    title_tfidf = pickle.load(file)  # dump information to that file
    file.close()

    # loading tfidf_body vectorizer
    file2 = open('body_tfidf.pickle', 'rb')
    body_tfidf = pickle.load(file2)
    file2.close()

    # opening model pickle
    file3 = open('model.pkl', 'rb')
    c_model = pickle.load(file3)
    file3.close()


    def output_pipeline(title, body):

        """
        Given the article title and body, this function returns if the news article is fake or not

        Parameters:
            title: The title of the article
            body: The body of the article

        Output:
            The function returns if the news is fake or not

        """
        stemmer = SnowballStemmer('english')

        # cleaning the title and body
        cleaned_title = []
        for line1 in title:

            cleaned_line1 = decontracted(line1)
            cleaned_line1 = deEmojify(cleaned_line1)  # Removing any emojis
            cleaned_line1 = cleaned_line1.replace('\n', ' ').replace('\t',
                                                                     ' ')  # Replacing any tab or new line indicators
            cleaned_line1 = re.sub('[^A-Za-z0-9]+', ' ', cleaned_line1)
            cleaned_line1 = ' '.join(
                word for word in cleaned_line1.split() if word.lower() not in stop_words)  # Removing stop words
            cleaned_line1 = cleaned_line1.lower().strip()  # Append the cleaned text and convert all text to lower case letters and remove any whitespace

            stemmed_title = []
            for word in cleaned_line1.split():
                stemmed_title.append(stemmer.stem(word))
            cleaned_title.append(' '.join(stemmed_title))

        cleaned_body = []
        for line2 in body:

            cleaned_line2 = decontracted(line2)
            cleaned_line2 = deEmojify(cleaned_line2)  # Removing any emojis
            cleaned_line2 = cleaned_line2.replace('\n', ' ').replace('\t',
                                                                     ' ')  # Replacing any tab or new line indicators
            cleaned_line2 = re.sub('[^A-Za-z0-9]+', ' ', cleaned_line2)
            cleaned_line2 = ' '.join(
                word for word in cleaned_line2.split() if word.lower() not in stop_words)  # Removing stop words
            cleaned_line2 = cleaned_line2.lower().strip()  # Append the cleaned text and convert all text to lower case letters and remove any whitespace

            stemmed_body = []
            for word in cleaned_line2.split():
                stemmed_body.append(stemmer.stem(word))
            cleaned_body.append(' '.join(stemmed_body))

        # Extracting features using tfidf from cleaned title and body
        X_title = title_tfidf.transform(cleaned_title)
        X_body = body_tfidf.transform(cleaned_body)

        # Combining the title and body features into one to pass through the model
        X = hstack((X_title, X_body))

        # Passing through the trained model and predicting
        res = c_model.predict(X)
        prob = c_model.predict_proba(X)

        return res, prob


    st.set_page_config(layout='wide')

    st.title('Fake-News Detection')
    st.write("""
    ## Is the news you're reading genuine or fake?
    """)
    st.markdown('<style>body{background-color: pink;}</style>', unsafe_allow_html=True)

    df = pd.read_csv('data//test.csv')
    data = df.head(50)

    nav = st.sidebar.radio('Go to', ['Home', 'Prediction', 'Contribute'])
    if nav == 'Home':
        st.write('### Go to the prediction tab to find out!')
        st.image('http://zignallabs.com/wp-content/uploads/2017/04/fake-news.jpg')
        st.write('What our dataset looks like')
        if st.checkbox('Show data'):
            st.dataframe(data, width = 1000)

    if nav == 'Prediction':
        st.image('https://ichef.bbci.co.uk/news/976/cpsprodpb/089D/production/_111750220_gettyimages-1215064495.jpg')
        st.write('### Find out here!')
        t = list(st.text_input('Enter the article title'))
        b = list(st.text_input('Enter the article body'))
        t = [''.join(t)]
        b = [''.join(b)]

        if st.button('Predict'):
            result, proba = output_pipeline(t, b)
            proba = proba*100
            if result == 0:
                md_results = f"This news article is **{proba[0][0]:.2f}%** genuine."
                st.markdown(md_results)
                #st.write('This article is {}% genuine',proba[0][0])
                #st.write(proba[0][0])
            else:
                md_results = f"This article is **{proba[0][1]:.2f}%** not genuine."
                st.markdown(md_results)
                #st.write('This article is {}% not genuine', proba[0][1])
                #st.write(proba[0][1])

    if nav == 'Contribute':
        st.image('https://res.cloudinary.com/people-matters/image/upload/q_auto,f_auto/v1572048856/1572048855.jpg')
        st.header('Contribute to our dataset')
        contribute_title = st.text_input('Enter the article title')
        contribute_body = st.text_input('Enter the article body')
        contribute_label = st.text_input('Enter fake or not')
        if st.button('Contribute'):
            pass
