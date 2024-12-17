import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB   # Not using this
from sklearn.svm import SVC
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity

# Download all requirements once
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


# Step 1: Dataset Preparation
dataset = [
    ("What is your name?", "My name is Chatbot."),
    ("How are you?", "I am doing well."),
    ("What is the weather today?", "The weather is sunny."),
    ("How old are you?", "I am an AI, so I don't have an age."),
    ("Where are you from?", "I exist in the digital world, so I don't have a physical location."),
    ("What is the meaning of life?", "The meaning of life can be subjective and varies for each individual."),
    ("Tell me a joke.", "Sure, here's one: Why don't scientists trust atoms? Because they make up everything!"),
    ("What is your favorite color?", "As an AI, I don't have personal preferences."),
    ("Can you sing a song?", "I'm afraid I don't have a singing voice."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the largest country in the world?", "The largest country in the world by land area is Russia."),
    ("Who is the current president of the United States?", "As of my knowledge cutoff in September 2023, the current president of the United States is Joe Biden."),
    ("What is the square root of 16?", "The square root of 16 is 4."),
    ("How do I reset my password?", "To reset your password, please follow the instructions provided by the platform or service you are using."),
    ("What is the best programming language?", "The best programming language depends on your needs and the specific task you want to accomplish."),
    ("Can you recommend a good book?", "Sure! One highly recommended book is 'Sapiens: A Brief History of Humankind' by Yuval Noah Harari."),
    ("How do I improve my communication skills?", "To improve your communication skills, you can practice active listening, seek feedback, and engage in regular conversations."),
    ("What is the latest technology trend?", "Artificial intelligence, machine learning, and blockchain are some of the latest technology trends."),
    ("How can I stay motivated?", "To stay motivated, you can set goals, break them into smaller tasks, celebrate achievements, and surround yourself with positive influences."),
    ("What is the best way to learn a new language?", "The best way to learn a new language is through consistent practice, immersion, and using various resources like language learning apps, books, and language exchange programs."),
    ("What are some healthy eating tips?", "Some healthy eating tips include consuming a balanced diet with plenty of fruits and vegetables, drinking enough water, and limiting processed foods and added sugars."),
    ("How can I manage stress?", "To manage stress, you can try relaxation techniques like deep breathing, exercise regularly, get enough sleep, and prioritize self-care."),
    ("What is the best way to start a small business?", "The best way to start a small business is by conducting market research, creating a business plan, securing funding, and seeking professional advice."),
    ("What is data science?", "Data science is a multidisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data."),
    ("What are the key skills for a data scientist?", "Key skills for a data scientist include programming (e.g., Python, R), statistical analysis, machine learning, data visualization, and domain knowledge in the specific field."),
    ("What is the role of a data scientist?", "The role of a data scientist is to collect, analyze, and interpret large and complex datasets to identify patterns, extract insights, and solve business problems."),
    ("What is machine learning?", "Machine learning is a subfield of artificial intelligence that focuses on the development of algorithms and models that enable computers to learn and make predictions or decisions without being explicitly programmed."),
    ("What is the difference between supervised and unsupervised learning?", "In supervised learning, the algorithm learns from labeled data where the input and the corresponding output are provided. In unsupervised learning, the algorithm learns from unlabeled data and finds patterns or structures in the data."),
    ("What is the purpose of data preprocessing?", "Data preprocessing is an essential step in data analysis and machine learning. It involves transforming raw data into a format suitable for analysis, cleaning data, handling missing values, encoding categorical variables, and scaling features."),
    ("What is the difference between correlation and causation?", "Correlation measures the statistical relationship between two variables, while causation implies a cause-and-effect relationship, meaning that changes in one variable directly cause changes in the other."),
    ("What is overfitting in machine learning?", "Overfitting occurs when a machine learning model performs well on the training data but fails to generalize to new, unseen data. It happens when the model becomes too complex and starts to memorize the noise or outliers in the training data."),
    ("What is cross-validation?", "Cross-validation is a technique used to evaluate the performance of a machine learning model. It involves dividing the data into multiple subsets, training the model on some subsets, and testing it on the remaining subset. This helps to assess how well the model generalizes to unseen data."),
    ("What is the difference between classification and regression?", "Classification is a task of predicting discrete labels or categories, while regression is a task of predicting continuous numerical values."), ("What is Natural Language Processing (NLP)?", "Natural Language Processing (NLP) is a field of study that focuses on the interaction between computers and human language. It involves tasks such as text classification, sentiment analysis, language translation, and information extraction."),
    ("What are the key steps in NLP?", "The key steps in NLP include tokenization, part-of-speech tagging, parsing, named entity recognition, semantic analysis, and discourse analysis. These steps help in understanding the structure and meaning of text."),
    ("What is tokenization?", "Tokenization is the process of breaking text into smaller units called tokens. Tokens can be individual words, phrases, or even characters depending on the task at hand."),
    ("What is part-of-speech tagging?", "Part-of-speech tagging is the process of assigning grammatical tags (such as noun, verb, adjective, etc.) to each word in a text. It helps in understanding the syntactic structure of sentences."),
    ("What is parsing?", "Parsing is the process of analyzing the grammatical structure of a sentence to determine its constituent parts and how they relate to each other. It helps in understanding the relationships between words."),
    ("What is named entity recognition (NER)?", "Named entity recognition is the process of identifying and classifying named entities (such as person names, organization names, locations, etc.) in text. It helps in extracting important information from text."),
    ("What is sentiment analysis?", "Sentiment analysis is the process of determining the sentiment or opinion expressed in a piece of text. It can classify text as positive, negative, or neutral, and is often used to analyze customer feedback, social media posts, and reviews."),
    ("What is language translation?", "Language translation is the task of converting text from one language to another while preserving the meaning. It involves understanding the structure and semantics of the source language and generating equivalent text in the target language."),
    ("What is information extraction?", "Information extraction is the process of automatically extracting structured information from unstructured text. It involves identifying and extracting specific types of information, such as names, dates, locations, and relationships, from text."),
    ("What is text summarization?", "Text summarization is the task of generating a concise and coherent summary of a longer text. It aims to capture the main points and key information while maintaining the overall meaning."),
    ("What are language models?", "Language models are statistical models that learn the patterns and structures of a language from a large corpus of text. They are used to generate new text, predict the next word in a sequence, and evaluate the probability of a given sequence of words."),
    ("What is word embedding?", "Word embedding is a technique used to represent words as dense vectors in a high-dimensional space. It captures the semantic relationships between words, allowing algorithms to better understand and process natural language."),
    ("What is topic modeling?", "Topic modeling is a technique used to discover hidden topics or themes in a collection of documents. It automatically identifies the main topics and their distribution in the documents."),
    ("What is named entity disambiguation?", "Named entity disambiguation is the process of determining the correct meaning or entity referred to by a given named entity. It helps in resolving ambiguities that arise due to entities with the same name."),
    ("What is text classification?", "Text classification is the task of categorizing text into predefined classes or categories. It is commonly used for sentiment analysis, spam detection, topic classification, and document classification."),
    ("What is information retrieval?", "Information retrieval is the task of retrieving relevant information from a large collection of unstructured or semi-structured text. It involves techniques such as indexing, querying, and ranking."),
    ("What is text normalization?", "Text normalization is the process of converting text into a standardized or canonical form. It involves tasks such as removing punctuation, converting to lowercase, handling contractions, and normalizing numbers."),
    ("What is text generation?", "Text generation is the task of automatically generating coherent and meaningful text. It can be used for various applications, such as chatbots, language translation, and content generation."),
    ("What are the challenges in NLP?", "Challenges in NLP include dealing with ambiguity, understanding context, handling different languages and dialects, handling noisy and informal text, and capturing complex linguistic phenomena.")
]

# Step 2: Preprocessing Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

preprocessed_dataset = [(preprocess_text(question), response) for question, response in dataset]

# Step 3: Create training data
X_train = [question for question, _ in preprocessed_dataset]
y_train = [response for _, response in preprocessed_dataset]

# Step 4: Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

# Step 5: Train the model (Naive Bayes) // not using this model now
# model = MultinomialNB()

# Step 5: Train the model (Support Vector Machine)
model = SVC(probability=True) # Use the SVC classifier
model.fit(X_train_vectors, y_train)

# Step 6: Evaluate the model
X_train_vectors = vectorizer.transform(X_train)
train_accuracy = model.score(X_train_vectors, y_train)
print("Model Accuracy: {:.2f}".format(train_accuracy))

# Step 7: Integrate with chatbot interface
user_name = input("Please enter your name: ")
chatbot_name = input("Please name me!\nMy Name: ")
print(chatbot_name + ": Hi", user_name + "! How can I assist you today?")


# Step 8: Chatbot Loop
while True:
    user_input = input(user_name + ": ")
    preprocessed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([preprocessed_input])
    predicted_response = model.predict(input_vector)[0]

    # Calculate cosine similarity for confidence level
    similarities = cosine_similarity(input_vector, X_train_vectors)
    max_similarity = similarities.max()
    confidence_level = max_similarity.item()
    # print(confidence_level)

    if confidence_level < 0.4:
        print(chatbot_name + ": As an AI, I don't have an answer for that. Please send an email to support@example.com")
    else:
        print(chatbot_name + ":", predicted_response)
    
    additional_question = input(chatbot_name + ": Is there anything else you want to know, (yes/no)?\n" + user_name + ": ")
    if additional_question.lower() == "no":
        print(chatbot_name + ": Okay, " + user_name + ", have a great day!")
        # Ask for user feedback
        feedback = input(chatbot_name + ": Please provide your feedback\n" + user_name + ": ")
        # Sentiment analysis
        sentiment = TextBlob(feedback).sentiment
        polarity = sentiment.polarity
        if polarity > 0:
            print(chatbot_name + ": I'm glad you found my response helpful, " + user_name + "!")
            break
        elif polarity < 0:
            print(chatbot_name + ": I apologize if my response was not satisfactory, " + user_name + " could you please provide feedback?")
            feedback_input = input(user_name + ": ")
            print(chatbot_name + ": Thank you for your feedback!")
            break
        else:
            print(chatbot_name + ": Thank you for your feedback, " + user_name + "!")
            break
    elif additional_question.lower() == "yes":
        continue