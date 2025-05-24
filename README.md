# 📰 Fake News Detection Using Machine Learning

This project demonstrates how to detect fake news using several supervised machine learning models. It walks through the complete pipeline — from data loading and preprocessing to model training, evaluation, and saving.

---

## 📊 Dataset

The dataset used in this project includes:

- A **text** column containing the full content of news articles.
- A **label** column indicating whether the news is `FAKE` or `REAL`.

The dataset is loaded from a CSV file and preprocessed before feeding into machine learning models.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Scikit-learn
- Pandas
- NumPy

---

## 🔍 Project Workflow

### 1. **Importing Libraries**
Essential libraries for data handling, modeling, and evaluation are imported.

### 2. **Loading the Dataset**
The CSV dataset is loaded using `pandas.read_csv()` and the first few entries are displayed for inspection.

### 3. **Data Preprocessing**
- Missing values are checked and removed.
- Feature (`text`) and target (`label`) columns are defined.
- The target column is label-encoded for compatibility with scikit-learn.

### 4. **Text Vectorization**
- The `CountVectorizer` is used to convert textual data into a bag-of-words representation.

### 5. **Train-Test Split**
- The dataset is split into training and testing sets (80/20 split) using `train_test_split`.

### 6. **Model Training and Evaluation**
The following machine learning models were implemented, trained, and evaluated:

- ✅ **Multinomial Naive Bayes**
- ✅ **Logistic Regression**
- ✅ **Support Vector Classifier (Linear Kernel)**
- ✅ **Decision Tree Classifier**
- ✅ **Passive Aggressive Classifier**

Each model is evaluated using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)

### 7. **Model Saving**
- The trained Naive Bayes model is saved using `joblib` for future reuse.

---

## 📈 Results

Each model's performance is printed in the notebook, including accuracy and a detailed classification report. This allows for easy comparison of how different models handle the fake news detection task.

---

## 💾 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Fake-News-Detection-ML-Models.git
   cd Fake-News-Detection-ML-Models
---

## 📌 Future Improvements
Use more advanced NLP techniques (TF-IDF, word embeddings).

- Deploy the model using Flask or Streamlit.

- Add visual comparisons of model performance.

- Incorporate deep learning models like LSTM or BERT.

  ## ✍Author
 **Mahbuba AKhtar Jidni**
