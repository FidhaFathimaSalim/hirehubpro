import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JobRecommender:
    def __init__(self, job_data):
        self.job_data = job_data
        self.job_data['Required_Skills'] = self.job_data['Required_Skills'].fillna('')  # Replace NaN with empty string
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.job_data['Title'])

    def recommend_jobs(self, user_skills, required_skills, top_n=5):
        if not user_skills.strip():
            print("Error: User skills input is empty.")
            return pd.DataFrame()
        if not required_skills.strip():
            print("Error: Required skills input is empty.")
            return pd.DataFrame()

        # Combine both user skills and required skills to form a query
        query = f"{user_skills} {required_skills}"

        # Vectorize the input query
        query_vec = self.vectorizer.transform([query])

        # Compute the cosine similarity between the query and the job titles
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix)

        # Get the top N most similar job indices
        top_indices = cosine_sim[0].argsort()[-top_n:][::-1]

        # Fetch the recommended jobs
        recommended_jobs = self.job_data.iloc[top_indices]
        return recommended_jobs[['Title', 'Required_Skills', 'Job_Description', 'URL']]

    @staticmethod
    def load_data(file_path):
        try:
            job_data = pd.read_csv(file_path)
            return job_data
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
            exit(1)
        except pd.errors.EmptyDataError:
            print("Error: The provided CSV file is empty.")
            exit(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            exit(1)

# Load the data
file_path = 'C:\\Users\\huawei\\Desktop\\hirehub\\hirehub\\jobportal\\job_dataset.csv'  # Replace with your actual file path
job_data = JobRecommender.load_data(file_path)

# Create an instance of the recommender
recommender = JobRecommender(job_data)

# Save the recommender model
joblib.dump(recommender, 'job_recommender.joblib')

# Load the recommender model
loaded_recommender = joblib.load('job_recommender.joblib')

# Loop to allow multiple searches
while True:
    # Example usage
    user_skills_input = input("Enter your skills (comma-separated): ")
    user_skills = ' '.join(user_skills_input.split(',')).strip()
    required_skills_input = input("Enter the required skills for the job (comma-separated): ")
    required_skills = ' '.join(required_skills_input.split(',')).strip()

    # Call the recommendation function
    recommended_jobs = loaded_recommender.recommend_jobs(user_skills, required_skills)

    # Display the recommendations
    if not recommended_jobs.empty:
        print("\nRecommended Jobs:")
        for _, row in recommended_jobs.iterrows():
            print(f"Title: {row['Title']}")
            print(f"Link: {row['URL']}\n")
    else:
        print("No jobs found matching your input.")

    # Ask if the user wants to make another search
    choice = input("\nDo you want to make another search? (yes/no): ").strip().lower()
    if choice not in ['yes', 'y']:
        print("Thank you for using the Job Recommendation System. Goodbye!")
        break