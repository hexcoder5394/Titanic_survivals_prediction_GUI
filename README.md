# Titanic_survivals_prediction_GUI
GUI Application for predict if the user survived from the titanic tragedy. 

# Titanic Survival Prediction Model

This project implements a machine learning model to predict whether a passenger survived the Titanic disaster based on their demographic and ticket information. The model is built using ensemble techniques combining RandomForestClassifier, GradientBoostingClassifier, and LogisticRegression algorithms.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- mysql-connector-python
- tkinter

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/titanic-survival-prediction.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have MySQL installed and running on your machine.

2. Set up the database:

    - Create a database named `titanicDB`.
    - Import the provided SQL dump file `titanic.sql` to create necessary tables and populate them with data.

3. Update database connection details:

    - Open `titanic_prediction.py` file.
    - Modify the `db` variable to include your MySQL connection details (host, database, user, password).

4. Run the application:

    ```bash
    python titanic_prediction.py
    ```

5. The application window will appear. Enter passenger details (Name, Age, Passenger Class, Sex, Fare) and click the "Check" button to see the prediction.

## Directory Structure

- `titanic_prediction.py`: Main Python script containing the machine learning model and Tkinter GUI.
- `requirements.txt`: File listing all Python dependencies.
- `titanic.sql`: SQL dump file containing database schema and sample data.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

