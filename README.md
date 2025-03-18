# Financial Sentiment Analysis for News Headlines

This repository contains the source code for my Master's Thesis project titled **"Design of a Machine Learning Model for Sentiment Analysis in Financial News."** The project addresses two main challenges:
1. **Extraction of the Main Economic Subject (POE):** Automatically identifying the principal economic entity in a financial news headline.
2. **Sentiment Analysis:** Determining the sentiment (positive, neutral, or negative) expressed in the headline toward the POE, as well as toward companies and consumers.

The project combines traditional NLP techniques (using NLTK and spaCy), classical machine learning models (e.g., Random Forest), large language models (LLMs) such as OpenAI's GPT-3, and hybrid approaches (e.g., spaCy + BERT).

## Project Overview

Financial news and social media data are increasingly used to predict stock market trends. Given the complexity of financial language and the context-dependent nature of sentiment, this work explores different models and preprocessing techniques to achieve robust sentiment analysis from financial headlines.

### Key Objectives
- **Main Economic Subject Extraction:** Automatically identify the principal economic subject (POE) in each headline.
- **Targeted Sentiment Analysis:** Classify the sentiment expressed in headlines toward:
  - The POE
  - Companies
  - Consumers

### Methodology
The project follows the CRISP-DM methodology, which includes:
- Business understanding and data comprehension
- Data preparation and preprocessing (using classical NLP libraries and transformer-based models)
- Model building using various approaches (NLTK, spaCy, OpenAI, spaCy+BERT, and hybrid methods)
- Evaluation using metrics such as accuracy and weighted F1-score, along with confusion matrices for detailed error analysis
- Analysis of feature importances and model explainability using tools like LIME

## Repository Structure

The repository (located at [https://github.com/runciter2078/tfm](https://github.com/runciter2078/tfm)) contains the following scripts:

- **explaning_lime.py**  
  Provides an example of model explainability using LIME. It demonstrates how to generate and display textual explanations for model predictions.

- **model_hybrid.py**  
  Implements a hybrid model that combines OpenAI's GPT-3 for POE extraction and sentiment analysis on financial headlines with traditional machine learning (Random Forest) for classifying sentiment toward companies and consumers.

- **model_nltk.py**  
  Uses NLTK for text preprocessing and feature extraction. It extracts the main economic subject using frequency-based methods and builds classifiers (Random Forest) for sentiment analysis across different economic entities.

- **model_openai.py**  
  Uses the OpenAI API (GPT-3) to extract the POE from headlines and perform sentiment analysis. Given API costs, the script operates on a sample (e.g., 5% of the data) for demonstration purposes.

- **model_spacy.py**  
  Utilizes spaCy for preprocessing and builds a Random Forest classifier pipeline with TF-IDF features. It evaluates sentiment classification performance for the target, companies, and consumers.

- **model_spacy_bert.py**  
  Combines spaCy for text preprocessing with a pre-trained BERT model (using the `nlptown/bert-base-multilingual-uncased-sentiment` pipeline) to predict sentiment for the POE, providing a deeper contextual analysis.

- **Additional Scripts:**  
  The repository also contains scripts for explainability (using LIME) and for model evaluation and feature importance extraction.

## Data

The scripts use the dataset provided by the competition **"IBERLEF 2023 Task - FinancES: Financial Targeted Sentiment Analysis in Spanish"**, which is available in CSV format (e.g., `FinancES_phase_2_train_public.csv`). More details about the dataset and annotations can be found in the thesis document.

## Thesis Document

The complete Master's Thesis document is available as a PDF named **TFM_Financial_Sentiment_Analysis.pdf** in the repository. It includes detailed information on the methodology, experiments, results, and future work.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - scipy
  - matplotlib
  - seaborn
  - nltk
  - spacy
  - transformers
  - lime
  - python-docx
  - (and others as needed)

Make sure to install the Spanish language model for spaCy:
```bash
python -m spacy download es_core_news_sm
```

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/runciter2078/tfm.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd tfm
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If a requirements file is not provided, install manually the packages listed above.)*

## How to Run the Scripts

Each script is self-contained and addresses a specific part of the project. For example:

- **To run the LIME explanation example:**
  ```bash
  python explaning_lime.py
  ```
- **To run the hybrid model:**
  ```bash
  python model_hybrid.py
  ```
- **To run the NLTK-based model:**
  ```bash
  python model_nltk.py
  ```
- **To run the OpenAI-based model:**
  ```bash
  python model_openai.py
  ```
- **To run the spaCy-based model:**
  ```bash
  python model_spacy.py
  ```
- **To run the spaCy+BERT model:**
  ```bash
  python model_spacy_bert.py
  ```

Adjust file paths and API keys as necessary in the scripts.

## License

This project is distributed under the [MIT License](LICENSE).
