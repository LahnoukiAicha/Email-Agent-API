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
        self.SIMILARITY_THRESHOLD = 0.3  # Default similarity threshold

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
                else:
                    assignments.append({
                        'id_email': email['id_email'],
                        'solution': solution,
                        'Agent_ID': None,
                        'Agent_Name': "No available agent",
                        'Email_Score': email['Lead_Score'],
                        'Agent_Score': None
                    })
        return pd.DataFrame(assignments)

@app.route('/assign_emails', methods=['POST'])
def assign_emails():
    try:
        # Load agents data from request
        agent_data = request.get_json().get('agents', [])
        if not agent_data or not isinstance(agent_data, list):
            return jsonify({"error": "Invalid agent data provided. Expecting a list of agents."}), 400

        # Load agents
        matcher = SolutionEmailAgentMatcher()
        matcher.load_agents(agent_data)

        # Call the first API to fetch classified emails
        classification_api_url = os.getenv('CLASSIFICATION_API_URL', 'https://email-classification-api.onrender.com/classify-emails')
        print(f"Fetching classified emails from: {classification_api_url}")

        # Create a dummy input file for the first API
        dummy_csv = (
            "id_email;company_name;email;phone_number;location;industry;annual_revenue;engagement_score;email_opens;website_visits;subject;email_text\n"
            "E008;Brock-Torres;charles74@marquez.info;123456782;418 Garner Crescent, Port Katherineview, ND 61149;"
            "Ophthalmologist;$171,689,448;0.26;42;34;E-commerce application help;I wanna integrate einstien chat bot in my commerce application."
        )
        
        # Send POST request to the first API
        response = requests.post(classification_api_url, files={"file": ("input.csv", dummy_csv)}, timeout=15)
        
        # Log the response for debugging
        print("First API Response Status Code:", response.status_code)
        print("First API Response Content:", response.text)

        # Ensure the response is valid JSON
        try:
            response_data = response.json()
        except ValueError as e:
            return jsonify({"error": f"Invalid JSON response from first API: {str(e)}", "response": response.text}), 500

        if 'classified_emails' not in response_data or not response_data['classified_emails']:
            return jsonify({"error": "No classified emails returned from the first API"}), 500

        # Load classified emails into a DataFrame
        email_df = pd.DataFrame(response_data['classified_emails'])

        # Assign emails to agents
        assignments = matcher.assign_emails_to_agents(email_df)

        print(f"Number of emails assigned: {len(assignments)}")
        return jsonify(assignments.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
