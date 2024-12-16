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

            # Filter agents to those who explicitly have the solution in their Skills
            priority_agents = available_agents[available_agents['Skills'].str.contains(solution, case=False, na=False)]

            # If we have agents that explicitly match the solution, use them; else use all available agents
            chosen_agents = priority_agents if not priority_agents.empty else available_agents

            for _, email in emails.iterrows():
                if not chosen_agents.empty:
                    # Sort by Agent_Score (original logic) and pick the top agent
                    agent = chosen_agents.sort_values(by='Agent_Score', ascending=False).iloc[0]
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
        data = request.get_json()
        agent_data = data.get('agents', [])
        classified_emails = data.get('classified_emails', [])

        if not agent_data or not isinstance(agent_data, list):
            return jsonify({"error": "No agents data provided or invalid format"}), 400

        if not classified_emails or not isinstance(classified_emails, list):
            return jsonify({"error": "No classified_emails data provided or invalid format"}), 400

        # Load agents into matcher
        matcher = SolutionEmailAgentMatcher()
        matcher.load_agents(agent_data)

        # Convert classified emails to a DataFrame
        email_df = pd.DataFrame(classified_emails)
        if email_df.empty:
            return jsonify({"error": "classified_emails data is empty"}), 400

        # Assign emails to agents
        assignments = matcher.assign_emails_to_agents(email_df)

        # Return assignments as JSON
        return jsonify(assignments.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5001)), debug=True)
