# Sentence-Autocompletion

This implementation builds a auto completion model using an LSTM neural network.  
It trains on a large text dataset ("Pride and Prejudice" by Jane Austen, sourced from Project Gutenberg) and predicts the next few words based on a given seed text.

The model learns word patterns and sequences through n-gram generation and sequence modeling techniques, ultimately enabling it to generate coherent text continuations.

1. **Download and Load Dataset**
   - Downloaded *Pride and Prejudice* text.
   - Loaded the dataset into memory.

2. **Data Preprocessing**
   - Removed unwanted characters (punctuation, numbers, etc.).
   - Converted the text to lowercase.
   - Split the text into sentences.

3. **Tokenization**
   - Converted sentences into sequences of word indices using a `Tokenizer`.
   - Created n-gram sequences to help the model learn word relationships.

4. **Padding Sequences**
   - Padded sequences to ensure uniform length for efficient training.

5. **Prepare Training Data**
   - Split sequences into:
     - **Input (X)**: All words except the last.
     - **Label (y)**: The last word (target to predict).
   - One-hot encoded the labels.

6. **Build the LSTM Model**
   - **Embedding Layer**: Turns words into 100-dimensional dense vectors.
   - **LSTM Layer**: Captures temporal relationships with 150 memory units.
   - **Dense Output Layer**: Uses `softmax` to predict the next word.

7. **Compile the Model**
   - Loss: `categorical_crossentropy` (multi-class classification).
   - Optimizer: `adam`.
   - Metric: `accuracy`.

8. **Predict Next Words**
   - A `predict_next_words()` function takes a seed text and predicts the next few words based on learned patterns.


## Example Usage
```python
seed_text = "I have been told about you, then"
next_words = 5
print(predict_next_words(seed_text, next_words, model, max_sequence_length))
```
This will extend your seed text by 5 predicted words.

