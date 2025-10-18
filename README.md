# Quora Duplicate Questions Detection (NLP Project)

Here you can find a complete deep learning project focused on **Natural Language Processing (NLP)**.  
The goal of this project is to detect whether two questions from the Quora platform have the same meaning (i.e., identify duplicate questions).  

This problem is a classic **semantic similarity** challenge, where models learn to understand relationships between text sequences.

The project has three main parts:

1. **Text Preprocessing and Tokenization**  
2. **Embedding and Feature Representation**  
3. **LSTM and other Model Training and Evaluation**

---

## 🧩 Stack

**Pandas**, **NumPy**, **Matplotlib**, **Scikit-Learn**, **TensorFlow**, **TensorFlow Hub**

---

## 📊 Text Preprocessing and Data Exploration

In this phase, I explored the text dataset containing pairs of questions.  
The preprocessing pipeline included:

- Lowercasing and punctuation removal (`string.punctuation`)  
- Tokenization using TensorFlow / Keras **Tokenizer**  
- Padding sequences to a fixed length using `pad_sequences()`  
- Optional vectorization using **TF-IDF**  

This ensured uniform input shapes for the neural model and reduced noise in the textual data.

### Example visualizations (placeholders):

- **Question length distribution**  
  <img width="505" height="340" alt="image" src="https://github.com/user-attachments/assets/da5f6cac-ee30-4eb6-a00d-a539f9bd8d20" />

  
- **Duplicate vs Non-Duplicate Ratio**  
<img width="762" height="382" alt="image" src="https://github.com/user-attachments/assets/7a40a2ce-d64d-4fce-b57b-e2079e6827d9" />

---

## 🧠 Embedding and Feature Representation

Word-level semantic meaning was captured through embeddings.  
Two approaches were used:

- **TF-IDF representation** for baseline models and vector inspection  
- **Embedding layer** in TensorFlow/Keras for deep learning models  
- Optionally, **pretrained embeddings** (e.g., from **TensorFlow Hub**) to enhance semantic understanding  

## Examples of Experiments: 
1. `model_0` Pipeline is baseline model_0 which give us a fundamentals on the beginning of experiments
2. `model_1 ` `text_vectorizer` with `text\token Embedding` and use `Conv1D` that model is worse that baseline
3. `model_2` use `tf_hub_sentence_encoder`
4. `model_3` use `char_vectorizer` with `char_embed` and use `Conv1D`
5. `model_4` combine `Token + Char embedding`

## results: 

<img width="821" height="700" alt="image" src="https://github.com/user-attachments/assets/1d3977a8-4b13-47f9-9d0d-87fd7b3d954c" />




