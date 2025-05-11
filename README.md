📘 Text Operation Assignment
This Python-based project is a text processing pipeline designed to analyze large collections of text documents. It includes all key steps in text preprocessing—such as normalization, tokenization, and stemming—and applies Luhn's cut-off to identify significant terms. The output includes indexed terms, frequency distribution plots (Zipf's Law), and a comprehensive summary report.

🔍 Features
Batch Corpus Loading
Loads and combines all .txt files from a specified folder efficiently.

Text Normalization
Converts all text to lowercase to ensure uniform processing.

Tokenization
Splits text into individual words while preserving:

Phone numbers

IP addresses

Dates

URLs

Stop Word Removal
Removes standard and custom stop words, while preserving essential words.

Stemming
Uses Porter Stemmer to reduce words to their root form.

Frequency Analysis
Counts word frequencies and applies Luhn's dynamic cut-off to filter significant terms.

Results Saving
Outputs results to a time-stamped folder, including:

Indexed terms

Terms before cut-off

Luhn cut-off points

Zipf's Law Plot
Generates a frequency-vs-rank plot visualizing term distribution.

Summary Report
Outputs key statistics from each processing step.

Group Member Listing
Prints information about the contributors.

💻 Requirements
Python 3.x

nltk

matplotlib

numpy

tqdm

Install all dependencies using:

bash
Copy
Edit
pip install nltk matplotlib numpy tqdm
📂 Usage
Prepare Your Corpus

Create a folder named corpus in the project directory.

Add your .txt files to this folder.

Create a file named stopword.txt with one stop word per line.

Run the Script

bash
Copy
Edit
python your_script_name.py
📁 Output will be saved in a folder like: output/indexed_terms_<timestamp>

📄 Output Includes:
before_luhns_cut.txt – All terms and their frequencies before cut-off.

final_indexed_terms.txt – Indexed terms after applying Luhn’s cut-off.

cutoff_points.txt – Lower and upper cut-off values.

frequency_vs_rank.png – Zipf’s Law frequency vs. rank plot.

summary_report.txt – Summary of processing statistics.

🛠️ Customization
Custom Stop Words
Add your own words to the custom_stopwords set in the script.

Essential Words
Add any important terms to the essential_words set to prevent their removal.

👥 Group Members

Berhanelidet Bekele – UGR/9452/16

Abel Engidu – UGR/5590/16

Dawit Temesgen – UGR/4848/16

Eldana Mulugeta – UGR/0191/16

Rodas Awgichew – UGR/8851/16

⚠️ Notes
The script is optimized for large corpora.

You can adjust or extend the pipeline for custom text analysis tasks.