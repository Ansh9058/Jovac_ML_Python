from flask import Flask, render_template, request
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import re
import nltk

nltk.download('punkt')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/text-summarization", methods=['POST'])
def summarize():
    if request.method == 'POST':
        inputtext = request.form['inputtext_']

        # Debugging: Print input text
        print("Input Text:", inputtext)

        # Clean up multiple spaces
        inputtext = str(re.sub(' +', ' ', inputtext))

        # Count number of sentences
        sentence_count = len(re.split(r'[.!?]+', inputtext))

        # Instantiate TextRankSummarizer
        summarizer = TextRankSummarizer()

        # Parse input text
        parser = PlaintextParser.from_string(inputtext, Tokenizer('english'))

        # Ensure at least 1 sentence is used in the summary
        num_sentences = max(1, round(0.2 * sentence_count))

        # Get summary sentences
        summary_sentences = summarizer(parser.document, num_sentences)

        # Build the summary string
        summary = ""
        for sentence in summary_sentences:
            summary += str(sentence)

        # Debugging: Print generated summary
        print("Summary:", summary)

    return render_template("output.html", data={'summary': summary})

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
