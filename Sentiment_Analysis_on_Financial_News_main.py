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

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load data
df = pd.read_csv('all-data.csv', header=None, encoding='latin1')

# Rename the unnamed column to 'article' for clarity
df.columns = ['label', 'article']

# Preprocessing function
def preprocess_text(text):
    """Preprocess text: remove non-alphabetic characters, tokenize, remove stopwords, and lemmatize."""
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I)
    text = text.lower()
    
    # Tokenization and stopword removal
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Apply preprocessing to the 'article' column
df['processed_text'] = df['article'].apply(preprocess_text)

# Check label distribution
print(df['label'].value_counts())  # Ensure labels are balanced

# Map labels to numeric values
y = df['label'].map({'positive': 1, 'negative': 0, 'neutral': 2})

# Split dataset into features and labels
X = df['processed_text']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences to ensure they are of the same length
max_sequence_length = 100  # Adjust this based on your dataset
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))  # 3 classes: positive, negative, neutral

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict sentiment function
def predict_sentiment(news_text):
    """Preprocess the input text and predict its sentiment."""
    # Preprocess the input text
    processed_text = preprocess_text(news_text)
    
    # Convert text to sequence and pad
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=max_sequence_length)
    
    # Predict sentiment
    pred = model.predict(padded_seq)
    
    # Get the label with the highest probability
    sentiment_labels = ['negative', 'positive', 'neutral']
    sentiment = sentiment_labels[pred.argmax()]
    
    return sentiment

# Create Tkinter UI
def create_ui():
    """Create the Tkinter user interface for sentiment analysis."""
    # Initialize the main Tkinter window
    window = tk.Tk()
    window.title("Financial News Sentiment Analysis")
    window.geometry("600x400")
    
    # Add instructions label
    instructions = tk.Label(window, text="Enter a Financial News Article to Analyze Sentiment", font=("Arial", 14))
    instructions.pack(pady=10)
    
    # Add a Text widget for user input
    text_input = tk.Text(window, height=10, width=70, font=("Arial", 12))
    text_input.pack(pady=10)
    
    # Add a label to display the result
    result_label = tk.Label(window, text="", font=("Arial", 14))
    result_label.pack(pady=10)
    
    def analyze_sentiment():
        """Analyze sentiment and update the result label."""
        news_text = text_input.get("1.0", tk.END).strip()  # Get text from text widget
        
        if not news_text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return
        
        # Get sentiment prediction
        sentiment = predict_sentiment(news_text)
        
        # Update result label
        result_label.config(text=f"Predicted Sentiment: {sentiment}")
    
    # Add a button to trigger sentiment analysis
    analyze_button = tk.Button(window, text="Analyze Sentiment", font=("Arial", 12), command=analyze_sentiment)
    analyze_button.pack(pady=10)
    
    # Run the Tkinter event loop
    window.mainloop()

# Run the user interface
create_ui()
