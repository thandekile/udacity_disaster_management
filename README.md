### Disaster Response Web Application

# Overview

This project builds a web application that categorizes disaster-related messages to aid emergency response teams. The app utilizes a trained machine learning model to automate message classification, improving disaster management efficiency and resource allocation.

# Setup Instructions

1) **Install Dependencies**
   Install the required Python packages by running:
   ```sh
   pip install pandas numpy sqlalchemy nltk scikit-learn flask plotly wordcloud joblib
   ```

2) **Prepare the Database and Train the Model**
   Execute the following scripts from the project root directory:

   - **Run ETL Pipeline**
     Cleans the data and stores it in a SQLite database.
     ```sh
     python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
     ```
   - **Run ML Pipeline**
     Trains the model and saves it as a pickle file.
     ```sh
     Please unzip the pickle before using tar -xf Name.zip
     python models/train_classifier.py models/classifier.pkl
     ```

3) **Start the Web Application**
   Launch the app with:
   ```sh
   python run.py
   ```
   Open the application in a web browser at:
   - https://lawknodlo5.prod.udacity-student-workspaces.com/

# Project Structure

```
project-root/
├── app/
│   ├── run.py                # Main Flask application
│   ├── templates/
│   │   ├── master.html       # Main webpage template
│   │   ├── go.html           # Classification result page
├── data/
│   ├── process_data.py       # ETL pipeline script
│   ├── disaster_messages.csv # Original messages dataset
│   ├── disaster_categories.csv # Original categories dataset
│   ├── DisasterResponse.db   # Processed database
├── models/
│   ├── train_classifier.py   # Machine learning pipeline script
│   ├── classifier.pkl        # Trained model
├── README.md                 # Documentation
```

# Web Application Features

The app provides interactive data visualizations to enhance disaster response insights:

- **Message Genre Distribution** – Displays message distribution across genres (e.g., direct, social, news).
- **Category Counts** – Shows the number of messages per category (e.g., related, request, offer).
- **Top 10 Disaster Categories** – Highlights the most frequent disaster-related message categories.
- **Most Used Words** – Presents a word cloud and bar chart of common words in the dataset.

# Machine Learning Pipeline

The pipeline employs GridSearchCV for hyperparameter tuning, ensuring high model accuracy. The trained classifier is saved as a pickle file (`classifier.pkl`) for seamless integration into the web application.

### Quick Start Guide:
1. **Prepare Data and Train Model:**
   ```sh
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```
2. **Navigate to the `app` directory:**
   ```sh
   cd app
   ```
3. **Launch the web application:**
   ```sh
   python run.py
   ```
4. **Click the `PREVIEW` button to access the homepage.**

# Contributing

Contributions are encouraged! Follow these steps to contribute:

1. **Fork the Repository** – Click the Fork button.
2. **Clone Your Fork** – Run:
   ```sh
   git clone https://github.com/your-username/disaster-response.git
   ```
3. **Create a New Branch** – Work on a separate branch:
   ```sh
   git checkout -b feature-branch
   ```
4. **Make Changes & Commit** – After making updates, commit with a meaningful message:
   ```sh
   git commit -m "Add feature X"
   ```
5. **Push & Submit a Pull Request** – Push the branch and create a pull request:
   ```sh
   git push origin feature-branch
   ```

### Contribution Guidelines
✔ Write clean, well-documented code.
✔ Ensure your changes do not break existing functionality.
✔ Clearly describe the purpose of your updates.

Let's collaborate to improve this project together!
