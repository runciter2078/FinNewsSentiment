# Financial Sentiment Analysis for News Headlines

This repository contains a collection of scripts for building a machine learning pipeline that analyzes sentiment in financial news headlines. The project focuses on two key tasks:

- **Main Economic Subject Extraction:** Automatically identifying the principal economic entity in each headline.
- **Targeted Sentiment Analysis:** Classifying the sentiment (positive, neutral, or negative) expressed toward both the main economic subject and related entities (e.g., companies and consumers).

The project integrates traditional NLP techniques (using NLTK and spaCy), classical machine learning models (such as Random Forest), and state-of-the-art language models (including OpenAI's GPT-3 and transformer-based approaches). In addition, hybrid methods (e.g., spaCy + BERT) are explored to enhance performance. The scripts are organized in a modular fashion with clear documentation, robust preprocessing, evaluation metrics, and model explainability routines (using LIME).

**Note:** This work was developed as part of my Master's Thesis (TFM) for the VIU (Universidad Internacional de Valencia) Big Data & Data Science program. Detailed documentation of the thesis is available in the file **TFM_Financial_Sentiment_Analysis.pdf**.

---

## Repository Structure

```
FinNewsSentiment/
├── explaning_lime.py       # Script demonstrating model explainability using LIME
├── model_hybrid.py         # Hybrid model combining OpenAI's GPT-3 for economic subject extraction and Random Forest for sentiment analysis
├── model_nltk.py           # NLTK-based preprocessing and Random Forest classifier for sentiment analysis
├── model_openai.py         # Script using the OpenAI API (GPT-3) for economic subject extraction and sentiment analysis
├── model_spacy.py          # spaCy-based pipeline with TF-IDF features and Random Forest classifier
├── model_spacy_bert.py     # Hybrid approach combining spaCy for preprocessing with a BERT-based sentiment classifier
├── README.md               # This file
├── requirements.txt        # List of required Python dependencies
└── TFM_Financial_Sentiment_Analysis.pdf  # Complete thesis document
```

---

## Requirements

- **Python:** 3.8 or above (Python 3.10+ recommended)
- **Dataset:** The project uses the CSV file `FinancES_phase_2_train_public.csv` (https://codalab.lisn.upsaclay.fr/competitions/10052).
- **Required Python Packages:**  
  Install via pip:
  ```bash
  pip install -r requirements.txt
  ```

*Additional note:* Ensure you have installed the Spanish language model for spaCy:
```bash
python -m spacy download es_core_news_sm
```

---

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/runciter2078/FinNewsSentiment.git
   cd FinNewsSentiment
   ```
2. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Keys and Data:**
   - Update the OpenAI API key within the scripts where needed.
   - Place the dataset CSV file (`FinancES_phase_2_train_public.csv`) in the repository folder or adjust the file path accordingly.
4. **Optional: Prepare a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## How to Run the Scripts

Each script is self-contained and can be executed independently. For example:
- **LIME Explanation Example:**
  ```bash
  python explaning_lime.py
  ```
- **Hybrid Model:**
  ```bash
  python model_hybrid.py
  ```
- **NLTK-based Model:**
  ```bash
  python model_nltk.py
  ```
- **OpenAI-based Model:**
  ```bash
  python model_openai.py
  ```
- **spaCy-based Model:**
  ```bash
  python model_spacy.py
  ```
- **spaCy+BERT Model:**
  ```bash
  python model_spacy_bert.py
  ```

---

## Evaluation and Results

Each script includes evaluation metrics (accuracy, weighted F1-score) and confusion matrices to assess model performance. Feature importance analysis and model explainability (using LIME) are also provided to help interpret the results.

---

## Additional Notes

- **Modularity:** The code is organized into independent modules, making it easy to update or extend.
- **Preprocessing:** Comprehensive text preprocessing is applied using both traditional NLP methods (NLTK) and modern libraries (spaCy).
- **Hybrid Approaches:** Various methods are combined to leverage the strengths of classical machine learning and transformer-based models.
- **Thesis Document:** For a detailed description of the methodology, experiments, and results, please refer to **TFM_Financial_Sentiment_Analysis.pdf**.

---

## License

This project is distributed under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements or bug fixes.

---
