import os
import re
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm

def load_corpus(folder_path):
    """
    Loads all text files from a folder and combines them into a single string.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    
    corpus = ""
    files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.txt')]
    
    for file_name in tqdm(files, desc="Loading Corpus"):
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8', errors='ignore') as file:
            corpus += file.read() + " "
    
    print(f"Processed {len(files)} file(s) from the folder '{folder_path}'.")
    return corpus

def normalize_text(text):
    """
    Normalizes the text by converting it to lowercase.
    """
    print("Normalizing Text...")
    normalized_text = ""
    for t in tqdm([text], desc="Normalizing Text"):
        normalized_text = t.lower()  # Convert the text to lowercase
    return normalized_text

def tokenize_text(text):
    """
    Tokenizes the text into individual words while removing punctuation and numerics,
    but preserving phone numbers, IP addresses, dates, and URLs.
    """
    print("Tokenizing Text...")

    # Define regex patterns to preserve specific formats
    phone_number_pattern = r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    ip_address_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'
    url_pattern = r'https?://(?:www\.)?\S+|www\.\S+'

    # Combine all patterns into one
    combined_pattern = f'({phone_number_pattern}|{ip_address_pattern}|{date_pattern}|{url_pattern})'

    # Find all matches for the patterns to preserve
    preserved_matches = re.findall(combined_pattern, text)

    # Remove all punctuation and numerics except for preserved patterns
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    cleaned_text = re.sub(r'\b\d+\b', ' ', cleaned_text)  # Remove standalone numbers

    # Tokenize the cleaned text with a progress bar
    tokens = []
    for word in tqdm(cleaned_text.split(), desc="Tokenizing Words"):
        tokens.append(word)

    # Add preserved matches back to the token list
    tokens.extend(preserved_matches)

    return tokens

def remove_stopwords(tokens, stopword_file, custom_stopwords=None, essential_words=None):
    """
    Removes stop words from the tokenized text using a stopword list from a file.
    """
    # Load stop words from the stopword.txt file
    try:
        with open(stopword_file, 'r', encoding='utf-8') as file:
            stop_words = set(file.read().splitlines())
    except FileNotFoundError:
        raise FileNotFoundError(f"The stopword file '{stopword_file}' does not exist.")

    # Add custom stop words if provided
    if custom_stopwords:
        stop_words = stop_words.union(set(custom_stopwords))

    # Filter tokens with a progress bar
    filtered_tokens = [
        word for word in tqdm(tokens, desc="Removing Stop Words")
        if word.lower() not in stop_words or word.lower() in (essential_words or [])
    ]

    # Calculate and print the number of removed stop words
    removed_count = len(tokens) - len(filtered_tokens)
    print(f"Removed {removed_count} stop words.")
    return filtered_tokens

def stem_words(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tqdm(tokens, desc="Stemming Words")]
    print(f"Stemmed {len(tokens)} words.")
    return stemmed_tokens

def calculate_frequencies(tokens):
    word_freq = Counter(tokens)
    return word_freq

def apply_dynamic_luhn_cutoff(word_freq):
    """
    Applies Luhn's dynamic cut-off to filter terms based on frequency percentiles.
    Returns a list of (word, frequency) tuples.
    """
    freqs = list(word_freq.values())
    lower_cutoff = np.percentile(freqs, 10)
    upper_cutoff = np.percentile(freqs, 95)
    filtered_terms = [
        (word, freq) for word, freq in tqdm(word_freq.items(), desc="Applying Luhn's Cut-Off")
        if lower_cutoff <= freq <= upper_cutoff
    ]
    print(f"Luhn's Dynamic Cut-Off Applied | Lower: {lower_cutoff:.2f}, Upper: {upper_cutoff:.2f}")
    return filtered_terms, lower_cutoff, upper_cutoff

def save_to_new_folder(terms, base_folder, lower_cutoff, upper_cutoff, timestamp, word_freq):
    """
    Saves indexed terms, terms before Luhn's cut-off, and cut-off points to a new folder.
    """
    new_folder = os.path.join(base_folder, f"indexed_terms_{timestamp}")
    os.makedirs(new_folder, exist_ok=True)

    # Save terms before Luhn's cut-off
    before_luhns_file_path = os.path.join(new_folder, 'before_luhns_cut.txt')
    try:
        with open(before_luhns_file_path, 'w', encoding='utf-8') as file:
            for word, freq in word_freq.items():
                file.write(f"{word}: {freq}\n")
        print(f"Terms before Luhn's cut-off successfully saved to: {before_luhns_file_path}")
    except Exception as e:
        print(f"Failed to save the terms before Luhn's cut-off file: {e}")

    # Save final indexed terms after Luhn's cut-off
    final_terms_file_path = os.path.join(new_folder, 'final_indexed_terms.txt')
    try:
        with open(final_terms_file_path, 'w', encoding='utf-8') as file:
            for word, freq in terms:
                file.write(f"{word}: {freq}\n")
        print(f"Final indexed terms successfully saved to: {final_terms_file_path}")
    except Exception as e:
        print(f"Failed to save the final indexed terms file: {e}")

    # Save cut-off points
    cutoffs_file_path = os.path.join(new_folder, 'cutoff_points.txt')
    try:
        with open(cutoffs_file_path, 'w', encoding='utf-8') as file:
            file.write(f"Lower Cut-Off: {lower_cutoff:.2f}\n")
            file.write(f"Upper Cut-Off: {upper_cutoff:.2f}\n")
        print(f"Cut-Off Points successfully saved to: {cutoffs_file_path}")
    except Exception as e:
        print(f"Failed to save the cut-off points file: {e}")

def plot_frequency_distribution(word_freq, output_folder):
    """
    Plots and saves the frequency vs rank distribution (Zipf's Law).
    """
    print("Plotting Frequency vs Rank (Zipf's Law)...")
    for _ in tqdm(range(1), desc="Plotting Frequency vs Rank"):
        frequencies = sorted(word_freq.values(), reverse=True)
        ranks = range(1, len(frequencies) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(ranks, frequencies, marker='o', linestyle='-', markersize=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.title("Frequency vs Rank (Zipf's Law)")
        plot_path = os.path.join(output_folder, 'frequency_vs_rank.png')
        plt.savefig(plot_path)
        print(f"Frequency vs Rank plot saved to: {plot_path}")
        plt.close()
def generate_summary_report(output_folder, total_words, stop_words_removed, words_stemmed, indexed_terms_count):
    """
    Generates a summary report and saves it to the output folder.
    """
    print("Generating Summary Report...")
    for _ in tqdm(range(1), desc="Generating Summary Report"):
        report_path = os.path.join(output_folder, 'summary_report.txt')
        try:
            with open(report_path, 'w', encoding='utf-8') as file:
                file.write(f"Total Words Processed: {total_words}\n")
                file.write(f"Stop Words Removed: {stop_words_removed}\n")
                file.write(f"Words Stemmed: {words_stemmed}\n")
                file.write(f"Indexed Terms: {indexed_terms_count}\n")
            print(f"Summary report saved to: {report_path}")
        except Exception as e:
            print(f"Failed to save summary report: {e}")

def print_group_members():
    print("\n--- Group Members ---")
    print(f"{'No.':<5}{'Group Member':<25}{'ID':<15}")
    print("-" * 45)
    print(f"{'1':<5}{'Berhanelidet Bekele':<25}{'UGR/9452/16':<15}")
    print(f"{'2':<5}{'Abel Engidu':<25}{'UGR/5590/16':<15}")
    print(f"{'3':<5}{'Dawit Temesgen':<25}{'UGR/4848/16':<15}")
    print(f"{'4':<5}{'Eldana Mulugeta':<25}{'UGR/0191/16':<15}")
    print(f"{'5':<5}{'Rodas Awgichew':<25}{'UGR/8851/16':<15}")
    print("-" * 45)

if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), "corpus")
    base_save_folder = os.path.join(os.getcwd(), "output")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist. Please create it and add .txt files.")
    if not os.path.exists(base_save_folder):
        os.makedirs(base_save_folder, exist_ok=True)

    print("Loading Corpus...")
    corpus = load_corpus(folder_path)
    print("Corpus Loaded!")

    print("Normalizing Text...")
    normalized_corpus = normalize_text(corpus)
    print("Text Normalized!")

    print("Tokenizing Text...")
    tokens = tokenize_text(normalized_corpus)
    print("Text Tokenized!")

    print("Removing Stop Words...")
    custom_stopwords = {"etc", "via", "thus", "therefore"}
    essential_words = {"not", "nor", "against"}
    stopword_file = os.path.join(os.getcwd(), "stopword.txt")
    tokens_no_stopwords = remove_stopwords(tokens, stopword_file, custom_stopwords, essential_words)
    print("Stop Words Removed!")

    print("Stemming Words...")
    stemmed_tokens = stem_words(tokens_no_stopwords)
    print("Words Stemmed!")

    print("Calculating Word Frequencies...")
    word_freq = calculate_frequencies(stemmed_tokens)
    print("Word Frequencies Calculated!")

    print("Applying Luhn's Cut-Off...")
    indexed_terms, lower_cutoff, upper_cutoff = apply_dynamic_luhn_cutoff(word_freq)
    print("Luhn's Cut-Off Applied!")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("Saving Indexed Terms and Cut-Off Points...")
    indexed_terms_folder = os.path.join(base_save_folder, f"indexed_terms_{timestamp}")
    save_to_new_folder(indexed_terms, base_save_folder, lower_cutoff, upper_cutoff, timestamp, word_freq)

    print("Plotting Frequency Distribution...")
    plot_frequency_distribution(word_freq, indexed_terms_folder)
    print("Frequency Distribution Plotted!")

    print("Generating Summary Report...")
    generate_summary_report(
        output_folder=indexed_terms_folder,
        total_words=len(tokens),
        stop_words_removed=(len(tokens) - len(tokens_no_stopwords)),
        words_stemmed=len(stemmed_tokens),
        indexed_terms_count=len(indexed_terms)
    )
    print("Summary Report Generated!")

    print_group_members()