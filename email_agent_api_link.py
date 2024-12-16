import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

class SolutionEmailAgentMatcher:
    def __init__(self):
        self.agent_vectorizer = None
        self.agent_df = None
        self.SIMILARITY_THRESHOLD = 0.3

    def load_agents(self, agent_data):
        self.agent_df = pd.DataFrame(agent_data)
        required_columns = ['Agent_ID', 'Agent_Name', 'Skills', 'Availability', 'Experience_Level', 'Performance_Score']
        if not all(col in self.agent_df.columns for col in required_columns):
            raise ValueError("Missing required columns in agent data")

        self.agent_vectorizer = TfidfVectorizer(stop_words='english')
        self.agent_vectorizer.fit(self.agent_df['Skills'])

        # Normalize and compute Agent_Score
        def normalize_experience(exp):
            return {'Junior': 33, 'Mid-Level': 66, 'Senior': 100}.get(exp, 0)

        self.agent_df['Agent_Score'] = (
            self.agent_df['Availability'].apply(lambda x: 100 if x else 0) * 0.4 +
            self.agent_df['Experience_Level'].apply(normalize_experience) * 0.3 +
            self.agent_df['Performance_Score'] * 0.3
        )

    def assign_emails_to_agents(self, email_df):
        assignments = []
        for solution, emails in email_df.groupby('Predicted_Solution'):
            agent_vectors = self.agent_vectorizer.transform(self.agent_df['Skills'])
            solution_vector = self.agent_vectorizer.transform([solution])
            similarity_scores = cosine_similarity(solution_vector, agent_vectors).flatten()
            available_agents = self.agent_df[similarity_scores > self.SIMILARITY_THRESHOLD].copy()

            for _, email in emails.iterrows():
                if not available_agents.empty:
                    agent = available_agents.sort_values(by='Agent_Score', ascending=False).iloc[0]
                    assignments.append({
                        'id_email': email['id_email'],
                        'solution': solution,
                        'Agent_ID': agent['Agent_ID'],
                        'Agent_Name': agent['Agent_Name'],
                        'Email_Score': email['Lead_Score'],
                        'Agent_Score': agent['Agent_Score']
                    })
        return pd.DataFrame(assignments)

@app.route('/assign_emails', methods=['POST'])
def assign_emails():
    try:
        # Load agents from request
        agent_data = request.get_json().get('agents', [])
        if not agent_data or not isinstance(agent_data, list):
            return jsonify({"error": "No agents data provided or invalid format"}), 400

        # Load agents into matcher
        matcher = SolutionEmailAgentMatcher()
        matcher.load_agents(agent_data)

        # Fetch classified emails from the first API
        classification_api_url = os.getenv('CLASSIFICATION_API_URL', 'https://email-classification-api.onrender.com/classify-emails')
        print(f"Fetching classified emails from: {classification_api_url}")

        # Call the first API
        response = requests.get(classification_api_url, timeout=60)
        print("First API Response Status Code:", response.status_code)
        print("First API Response Content:", response.text)

        if response.status_code != 200:
            return jsonify({"error": f"First API returned status {response.status_code}", "details": response.text}), 500

        response_data = response.json()
        if 'classified_emails' not in response_data or not response_data['classified_emails']:
            return jsonify({"error": "No classified emails found in the response"}), 500

        # Convert classified emails to a DataFrame
        email_df = pd.DataFrame(response_data['classified_emails'])

        # Assign emails to agents
        assignments = matcher.assign_emails_to_agents(email_df)

        # Return assignments as JSON
        return jsonify(assignments.to_dict(orient='records'))

    except requests.exceptions.Timeout:
        return jsonify({"error": "Request to the first API timed out"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch classified emails: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5001)), debug=True)
