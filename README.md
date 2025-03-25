Heart Failure Prediction using Machine Learning

Overview
This project uses a Random Forest classifier to predict heart failure based on a set of clinical features. The model is deployed using a FastAPI app, which serves predictions in real-time and stores past predictions in a SQLite database.

Repository Contents
- `heart_failure_model.ipynb`: Jupyter Notebook file containing the code for building and training the Random Forest classifier.
- `app.py`: Python file containing the FastAPI app code for serving predictions and interacting with the SQLite database.
- `model.pkl`: Pickled Random Forest classifier model file.

Requirements
- Python 3.8+
- FastAPI
- scikit-learn
- pandas
- sqlite3

Usage
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the FastAPI app using `uvicorn app:app --host 0.0.0.0 --port 8000`.
4. Use a tool like `curl` or a REST client to send POST requests to the app with JSON data containing the clinical features.
5. The app will return a prediction result indicating the likelihood of heart failure.

Contributing
Contributions are welcome! If you'd like to improve the model or add new features to the app, please fork the repository and submit a pull request.

Acknowledgments
- Appreciation to Kaggle for the dataset.
