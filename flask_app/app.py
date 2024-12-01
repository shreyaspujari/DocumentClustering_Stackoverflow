from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import os
import re
from html import unescape
from urllib.parse import unquote
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Flask app initialization
app = Flask(__name__)

# Utility function to clean text

def clean_text(text):
    """Remove HTML tags, decode HTML entities, and trim whitespace."""
    if not isinstance(text, str):
        return ""
    # Decode HTML entities (e.g., &amp; -> &)
    text = unescape(text)
    # Remove HTML tags (e.g., <p>...</p>)
    text = re.sub(r'<[^>]*>', '', text)
    # Normalize whitespace
    return text.strip()


# Directories
MODEL_DIR = '../models'
DATA_DIR = '../data'
STATIC_DIR = './static/wordclouds'

# Ensure the static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# Load clustering models and results
print("Loading clustering models and results...")

try:
    with open(f'{MODEL_DIR}/kmeans_model.pkl', 'rb') as file:
        kmeans_model = pickle.load(file)
    kmeans_labels = kmeans_model.labels_
except Exception as e:
    print(f"Error loading KMeans model: {e}")
    kmeans_model = None

try:
    with open(f'{MODEL_DIR}/dbscan_model.pkl', 'rb') as file:
        dbscan_labels = pickle.load(file)
except Exception as e:
    print(f"Error loading DBSCAN model: {e}")
    dbscan_labels = None

try:
    with open(f'{MODEL_DIR}/hdbscan_model.pkl', 'rb') as file:
        hdbscan_clusterer = pickle.load(file)
    hdbscan_labels = hdbscan_clusterer.labels_
except Exception as e:
    print(f"Error loading HDBSCAN model: {e}")
    hdbscan_clusterer = None

# Load PCA-reduced vectors for visualization
try:
    with open(f'{MODEL_DIR}/reduced_vectors.pkl', 'rb') as file:
        reduced_vectors = pickle.load(file)
except Exception as e:
    print(f"Error loading PCA-reduced vectors: {e}")
    reduced_vectors = None

# Load tag-based clustering results
try:
    with open(f'{MODEL_DIR}/tag_clusters.pkl', 'rb') as file:
        tag_clusters = pickle.load(file)
except Exception as e:
    print(f"Error loading tag-based clusters: {e}")
    tag_clusters = None

# Load the sentence transformer model
# Load the saved model, embeddings, and data
with open(f'{MODEL_DIR}/sbert_model.pkl', 'rb') as f:
    sbert_model = pickle.load(f)

with open(f'{MODEL_DIR}/sentence_embeddings.pkl', 'rb') as f:
    sentence_embeddings = pickle.load(f)

# Load processed dataset
try:
    final_data = pd.read_csv(f'{DATA_DIR}/final_dataset.csv')

    # Combine Question_Body and Answer_Body into a single 'Text' column if it doesn't already exist
    if 'Text' not in final_data.columns:
        final_data['Text'] = final_data['QuestionText'].astype(str) + " " + final_data['AnswerText'].astype(str)
except Exception as e:
    print(f"Error loading dataset: {e}")
    final_data = None

# Generate word clouds
def generate_word_cloud(cluster_text, cluster_name):
    if not cluster_text.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    image_path = f"{STATIC_DIR}/{cluster_name}.png"
    wordcloud.to_file(image_path)
    return image_path

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clusters')
def clusters():
    cluster_type = request.args.get('type', 'kmeans')
    cluster_number = int(request.args.get('cluster', 0))

    if cluster_type == 'kmeans' and kmeans_labels is not None:
        cluster_data = final_data[kmeans_labels == cluster_number]
    elif cluster_type == 'dbscan' and dbscan_labels is not None:
        cluster_data = final_data[dbscan_labels == cluster_number]
    elif cluster_type == 'hdbscan' and hdbscan_labels is not None:
        cluster_data = final_data[hdbscan_labels == cluster_number]
    else:
        return jsonify({"error": "Invalid cluster type or model not loaded"}), 400

    text = " ".join(cluster_data['Text'].dropna())
    wordcloud_path = generate_word_cloud(text, f"{cluster_type}_cluster_{cluster_number}")

    return render_template('clusters.html', cluster_type=cluster_type, cluster_number=cluster_number, wordcloud_path=wordcloud_path)

@app.route('/tags')
def tags():
    if not tag_clusters:
        return jsonify({"error": "Tag clustering results not found"}), 404

    return render_template('tags.html', tags=tag_clusters.keys())

@app.route('/tag/<tag>')
def tag_questions(tag):
    if tag not in tag_clusters:
        return jsonify({"error": f"Tag '{tag}' not found"}), 404
    
    clusters = tag_clusters[tag]
    # Zip questions and their cluster IDs in Python
    questions_and_clusters = list(zip([clean_text(q) for q in clusters['questions']], clusters['clusters']))
    max_cluster_id = max(cluster_id for _, cluster_id in questions_and_clusters)

    return render_template(
        'tag_questions.html',
        tag=tag,
        questions_and_clusters=questions_and_clusters,
        max_cluster_id=max_cluster_id
    )

@app.route('/question/<question>')
def question_answer(question):
    # Decode the question from the URL
    decoded_question = unquote(question)

    # Clean the decoded question
    cleaned_question = clean_text(decoded_question)

    # Find the matching answer for the cleaned question
    answer = final_data.loc[
        final_data['QuestionText'].apply(clean_text) == cleaned_question, 'AnswerText'
    ].values

    if len(answer) == 0:
        # If no matching answer is found, return a user-friendly message
        return render_template(
            'question_answer.html',
            question=cleaned_question,
            answer="Answer not found for this question."
        )

    # Clean the answer before passing it to the template
    cleaned_answer = clean_text(answer[0])

    return render_template('question_answer.html', question=cleaned_question, answer=cleaned_answer)

@app.route('/search', methods=['POST'])
def search():
    user_question = request.form.get('question')
    cleaned_user_question = clean_text(user_question)

    # Encode the user's question
    user_question_embedding = sbert_model.encode([cleaned_user_question], convert_to_tensor=True)

    # Compute cosine similarity
    similarities = cosine_similarity(user_question_embedding.cpu().detach().numpy(), sentence_embeddings.cpu().detach().numpy())
    most_similar_index = np.argmax(similarities)

    # Retrieve the corresponding question, answer, and tag
    best_match_question = final_data.iloc[most_similar_index]['QuestionText']
    best_match_answer = final_data.iloc[most_similar_index]['AnswerText']
    best_match_tag = final_data.iloc[most_similar_index]['Tag']

    return render_template(
        'search.html',
        user_question=user_question,
        predicted_answer=best_match_answer,
        best_match_question=best_match_question,
        best_match_tag=best_match_tag
    )


if __name__ == '__main__':
    app.run(debug=True)
