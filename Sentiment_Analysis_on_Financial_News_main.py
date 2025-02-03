import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import messagebox


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load data
df = pd.read_csv('all-data.csv', header=None, encoding='latin1')

df.columns = ['label', 'article']


def preprocess_text(text):
    """Preprocess text: remove non-alphabetic characters, tokenize, remove stopwords, and lemmatize."""
  
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I)
    text = text.lower()
    
   
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)


df['processed_text'] = df['article'].apply(preprocess_text)


print(df['label'].value_counts())  # Ensure labels are balanced


y = df['label'].map({'positive': 1, 'negative': 0, 'neutral': 2})

X = df['processed_text']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


max_sequence_length = 100  
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))  


model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))


test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

def predict_sentiment(news_text):
    """Preprocess the input text and predict its sentiment."""
   
    processed_text = preprocess_text(news_text)
    
 
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=max_sequence_length)
    

    pred = model.predict(padded_seq)
    
  
    sentiment_labels = ['negative', 'positive', 'neutral']
    sentiment = sentiment_labels[pred.argmax()]
    
    return sentiment

def create_ui():
    """Create the Tkinter user interface for sentiment analysis."""
    
    window = tk.Tk()
    window.title("Financial News Sentiment Analysis")
    window.geometry("600x400")
    
   
    instructions = tk.Label(window, text="Enter a Financial News Article to Analyze Sentiment", font=("Arial", 14))
    instructions.pack(pady=10)
    
  
    text_input = tk.Text(window, height=10, width=70, font=("Arial", 12))
    text_input.pack(pady=10)
    
   
    result_label = tk.Label(window, text="", font=("Arial", 14))
    result_label.pack(pady=10)
    
    def analyze_sentiment():
        """Analyze sentiment and update the result label."""
        news_text = text_input.get("1.0", tk.END).strip()  # Get text from text widget
        
        if not news_text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return
        
       
        sentiment = predict_sentiment(news_text)
        
      
        result_label.config(text=f"Predicted Sentiment: {sentiment}")
    
    analyze_button = tk.Button(window, text="Analyze Sentiment", font=("Arial", 12), command=analyze_sentiment)
    analyze_button.pack(pady=10)
   
    window.mainloop()

create_ui()
