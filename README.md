# Financial Sentiment Analysis for News Headlines

This repository contains a collection of scripts developed for the Master's Thesis project titled **"Design of a Machine Learning Model for Sentiment Analysis in Financial News."** The project focuses on analyzing financial news headlines by addressing two key tasks:

- **Main Economic Subject Extraction:** Automatically identifying the principal economic entity in each headline.
- **Targeted Sentiment Analysis:** Classifying the sentiment (positive, neutral, or negative) expressed in the headline toward the main economic subject, as well as toward companies and consumers.

The project combines traditional NLP techniques (using NLTK and spaCy), classical machine learning models (such as Random Forest), and state-of-the-art language models (including OpenAI's GPT-3 and transformer-based approaches). In addition, hybrid methods (e.g., spaCy + BERT) are explored to enhance performance.

The scripts in this repository have been developed with clarity and modularity in mind, featuring proper documentation, improved preprocessing, evaluation, and explainability routines (using LIME).  
 
**Note:** This work is part of my Master's Thesis (TFM), documented in detail in the thesis file **TFM_Financial_Sentiment_Analysis.pdf**.

---

## Repository Structure

```
FinNewsSentiment/
├── explaning_lime.py       # Script demonstrating model explainability using LIME
├── model_hybrid.py         # Hybrid model combining OpenAI's GPT-3 for POE extraction and Random Forest for sentiment analysis
├── model_nltk.py           # NLTK-based preprocessing and Random Forest classifier for sentiment analysis
├── model_openai.py         # Script using the OpenAI API (GPT-3) for economic subject extraction and sentiment analysis
├── model_spacy.py          # spaCy-based pipeline with TF-IDF features and Random Forest classifier
├── model_spacy_bert.py     # Hybrid approach combining spaCy for preprocessing with a BERT sentiment classifier
├── README.md               # This file
├── requirements.txt        # List of required Python dependencies
└── TFM_Financial_Sentiment_Analysis.pdf  # Complete thesis document
```

---

## Requirements

- **Python:** 3.8 or above (Python 3.10+ recommended)
- **Required Packages:**  
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

1. **Clone the repository:**
   ```bash
   git clone https://github.com/runciter2078/FinNewsSentiment.git
   cd FinNewsSentiment
   ```
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Keys and Data:**
   - Configure your OpenAI API key as needed inside the scripts.
   - The dataset used is provided in the CSV file `FinancES_phase_2_train_public.csv` (or adjust the path if es necesario).
4. **Run the scripts:**
   Each script is self-contained and can be executed independently. For example:
   - To run the LIME explanation example:
     ```bash
     python explaning_lime.py
     ```
   - To run the hybrid model:
     ```bash
     python model_hybrid.py
     ```
   - And so on.

---

## License

This project is distributed under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements or bug fixes.
