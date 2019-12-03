import pickle
import numpy
import tflearn
import tensorflow
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
from Stock import Stock


class chat:
    def __init__(self):
        self.model = []
        self.data = None
        self.stemmer = LancasterStemmer()
        self.words = []
        self.labels = []
        self.tick = None

    def model_load(self):
        nltk.download('punkt')
        with open('inputs/intents.json') as file:
            self. data = json.load(file)

        #print(self.data['intents'])

        try:
            with open('models/data.pickle', 'rb') as f:
                self.words, self.labels, training, output = pickle.load(f)
        except:
            docs_x = []
            docs_y = []

            for intent in self.data['intents']:
                for pattern in intent['patterns']:
                    wrds = nltk.word_tokenize(pattern)
                    self.words.extend(wrds)
                    docs_x.append(wrds)
                    docs_y.append(intent['tag'])

                    if intent['tag'] not in self.labels:
                        self.labels.append(intent['tag'])
            self.words = [self.stemmer.stem(w.lower()) for w in self.words if w != '?']
            self.words = sorted(list(set(self.words)))

            self.labels = sorted(self.labels)

            training = []
            output = []
            out_empty = [0 for _ in range(len(self.labels))]

            for x, doc in enumerate(docs_x):
                bag = []

                wrds = [self.stemmer.stem(w) for w in doc]

                for w in self.words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                output_row = out_empty[:] #to make a copy
                output_row[self.labels.index(docs_y[x])] = 1

                training.append(bag)
                output.append(output_row)

            training = numpy.array(training)
            output = numpy.array(output)
            with open('models/data.pickle', 'wb') as f:
                pickle.dump((self.words, self.labels, training, output), f)

        tensorflow.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation = 'softmax')
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)
        #print("training:",training)
        #print("/noutput:",output)
        try:
           self.model.load('models/model.tflearn')
        except:
           self.model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
           self.model.save("models/model.tflearn")

    def bag_of_words(self,s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, W in enumerate(words):
                if W == se:
                    bag[i] = 1

        return numpy.array(bag)

    def chatter(self,inp):
        results = self.model.predict([self.bag_of_words(inp, self.words)])[0]
        results_index = numpy.argmax(results)
        tag = self.labels[results_index]
        print(tag)
        print(results)
        if results[results_index] > 0.3:
            if tag == 'stock':
                stock_name = "#" + inp.split('#')[1]
                self.tick = stock_name
                res = str('What would you like to know about ' + stock_name.upper()
                          + '<br/>1)Stock Data '
                          + '<br/>2)Twitter Sentiment '
                          + '<br/>3)Investor Sentiment'
                          + '<br/>4)Predict Close Price'
                          + '<br/>5)Compare with another Stock')
                return res
            elif tag == 'historic_data':
                st = Stock(self.tick)
                return st.daily_stock_data()
            elif tag == 'twitter':
                st = Stock(self.tick)
                return st.tweet_Sentiment()
            elif tag == 'investor':
                st = Stock(self.tick)
                return st.investor_sentiment()
            elif tag == 'predict':
                st = Stock(self.tick)
                return st.stock_predict()
            elif tag == 'compare':
                st = Stock(self.tick)
                input = inp.split(' ')
                compare_stock = input[len(input)-1]
                return st.daily_stock_data(compare_stock)
            else:
                reply = None
                for tg in self.data['intents']:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                        #print( random.choice(responses))
                        reply = random.choice(responses)
                        return reply
        else:
            #print("I didn't get that. Please try again")
            reply = "I didn't get that. Please try again"
        return reply
def main():
    c = chat()
    c.model_load()
    print("Start chatting with the bot...(Hit Stop to quit)")
    while True:
        inp = input("You: ")
        if inp.lower() == 'stop':
            break
        print(c.chatter(inp))
if __name__ == '__main__':
    main()