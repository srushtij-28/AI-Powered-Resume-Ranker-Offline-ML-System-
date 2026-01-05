from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumes = [
    "Python developer with experience in data science and machine learning",
    "Java backend engineer with spring boot and mysql",
    "Data analyst skilled in python sql and visualization",
    "AI engineer with deep learning and neural networks"
]

job = "Looking for a python developer with machine learning and data analysis skills"

vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(resumes + [job])

scores = cosine_similarity(vectors[-1], vectors[:-1])[0]

ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print("📄 Resume Ranker \n")
for i, score in ranked:
    print(f"Resume {i+1} → Match Score: {score*100:.2f}%")
    print(resumes[i], "\n")
