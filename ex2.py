import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support

plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

#Read csv file
df = pd.read_csv('spam_or_not_spam.csv')

#We check whether there are missing values and we change them to 'Empty'
j=0;
q=[]
for i in range(0,len(df['email'])):
    if df['email'][i]==' ' or pd.isnull(df['email'][i])==True:
        j=j+1
        q.append(i)

df.at[q[0],'email'] = 'Empty'
df.at[q[1],'email'] = 'Empty'
df.at[q[2],'email'] = 'Empty'


sentences=df['email'].values
y = df['label'].values

#Train - Test split (75% and 25%)
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

#Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#Neural Network
embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
history = model.fit(X_train, y_train,
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
plt.show() #need to close it so that the program continues

results = model.predict(X_test)

#Metrics
results = np.around(results)
results = results.astype(int)

metric = precision_recall_fscore_support(y_test, results, average='macro')
print("\nPrecision: " + str(metric[0]))
print("Recall: " + str(metric[1]))
print("F1-Score: " + str(metric[2]))



