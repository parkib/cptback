# Backend for Medical Prediction Models

Welcome to the backend repository for our medical prediction models. This project provides the API and core logic for three machine learning models: stroke detection, heart attack prediction, and Titanic survival prediction. The backend handles data processing, model inference, and serves predictions to the frontend.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Overview

This backend application provides RESTful APIs for interacting with three different machine learning models:
1. **Stroke Detection**: Predicts the likelihood of a stroke based on user input data.
2. **Heart Attack Prediction**: Estimates the risk of a heart attack using relevant health metrics.
3. **Titanic Survival Prediction**: Predicts the chances of survival on the Titanic given certain passenger details.

## Features

- **RESTful API**: Provides endpoints for making predictions with the machine learning models.
- **Data Validation**: Ensures input data is valid and clean before making predictions.
- **Scalable Architecture**: Designed to handle multiple requests and scale efficiently.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (optional but recommended)

### Steps

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/medical-prediction-backend.git
    cd medical-prediction-backend
    ```

2. **Create a Virtual Environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Start the Server**:
    ```sh
    python app.py
    ```

The backend server should now be running on `http://localhost:5000`.

## Usage

The backend provides endpoints for making predictions with each model. Below are the details for each endpoint.

## API Endpoints

1. **Stroke Detection**:
    - **Endpoint**: `/predict/stroke`
    - **Method**: `POST`
    - **Request Body**:
        ```json
        {
            "age": 67,
            "hypertension": 0,
            "heart_disease": 1,
            "ever_married": "Yes",
            "work_type": "Private",
            "residence_type": "Urban",
            "avg_glucose_level": 228.69,
            "bmi": 36.6,
            "smoking_status": "formerly smoked"
        }
        ```
    - **Response**:
        ```json
        {
            "prediction": "No Stroke",
            "confidence": 0.85
        }
        ```

2. **Heart Attack Prediction**:
    - **Endpoint**: `/predict/heart-attack`
    - **Method**: `POST`
    - **Request Body**:
        ```json
        {
            "age": 57,
            "sex": 1,
            "cp": 0,
            "trestbps": 140,
            "chol": 192,
            "fbs": 0,
            "restecg": 1,
            "thalach": 148,
            "exang": 0,
            "oldpeak": 0.4
        }
        ```
    - **Response**:
        ```json
        {
            "prediction": "Low Risk",
            "confidence": 0.78
        }
        ```

3. **Titanic Survival Prediction**:
    - **Endpoint**: `/predict/titanic`
    - **Method**: `POST`
    - **Request Body**:
        ```json
        {
            "age": 29,
            "sex": "male",
            "pclass": 1,
            "siblings_spouses_aboard": 0,
            "parents_children_aboard": 0,
            "fare": 100
        }
        ```
    - **Response**:
        ```json
        {
            "prediction": "Survived",
            "confidence": 0.92
        }
        ```

## Model Details

### Stroke Detection Model

- **Input Features**: Age, Hypertension, Heart Disease, Ever Married, Work Type, Residence Type, Average Glucose Level, BMI, Smoking Status.
- **Algorithm**: Logistic Regression / Random Forest / Other (Specify the algorithm used).

### Heart Attack Prediction Model

- **Input Features**: Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Maximum Heart Rate Achieved, Exercise Induced Angina, ST Depression.
- **Algorithm**: Support Vector Machine / Decision Tree / Other (Specify the algorithm used).

### Titanic Survival Prediction Model

- **Input Features**: Age, Gender, Passenger Class, Siblings/Spouses Aboard, Parents/Children Aboard, Fare.
- **Algorithm**: Decision Tree / K-Nearest Neighbors / Other (Specify the algorithm used).

## Contributing

We welcome contributions to improve the project. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure your code follows our coding conventions and is well-documented.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to open issues or contact us if you have any questions or need further assistance. Happy coding!

### Files and Directories in this Project

These are some of the key files and directories in this project

README.md: This file contains instructions for setting up the necessary tools and cloning the project. A README file is a standard component of all properly set up GitHub projects.

requirements.txt: This file lists the dependencies required to turn this Python project into a Flask/Python project. It may also include other backend dependencies, such as dependencies for working with a database.

main.py: This Python source file is used to run the project. Running this file starts a Flask web server locally on localhost. During development, this is the file you use to run, test, and debug the project.

Dockerfile and docker-compose.yml: These files are used to run and test the project in a Docker container. They allow you to simulate the project’s deployment on a server, such as an AWS EC2 instance. Running these files helps ensure that your tools and dependencies work correctly on different machines.

instances: This directory is the standard location for storing data files that you want to remain on the server. For example, SQLite database files can be stored in this directory. Files stored in this location will persist after web application restart, everyting outside of instances will be recreated at restart.

static: This directory is the standard location for files that you want to be cached by the web server. It is typically used for image files (JPEG, PNG, etc.) or JavaScript files that remain constant during the execution of the web server.

api: This directory contains code that receives and responds to requests from external servers. It serves as the interface between the external world and the logic and code in the rest of the project.

model: This directory contains files that implement the backend functionality for many of the files in the api directory. For example, there may be files in the model directory that directly interact with the database.

templates: This directory contains files and subdirectories used to support the home and error pages of the website.

.gitignore: This file specifies elements to be excluded from version control. Files are excluded when they are derived and not considered part of the project’s original source. In the VSCode Explorer, you may notice some files appearing dimmed, indicating that they are intentionally excluded from version control based on the rules defined in .gitignore.

### Implementation Summary

#### July 2023

> Updates for 2023 to 2024 school year.

- Update README with File Descriptions (anatomy)
- Add JWT and add security features to data
- Add migrate.sh to support sqlite schema and data upgrade

#### January 2023

> This project focuses on being a Python backend server.  Intentions are to only have simple UIs an perhaps some Administrative UIs.

#### September 2021

> Basic UI elements were implemented showing server side Flask with Jinja 2 capabilities.

- Project entry point is main.py, this enables Flask Web App and provides capability to renders templates (HTML files)
- The main.py is the  Web Server Gateway Interface, essentially it contains a HTTP route and HTML file relationship.  The Python code constructs WSGI relationships for index, kangaroos, walruses, and hawkers.
- The project structure contains many directories and files.  The template directory (containing html files) and static directory (containing js files) are common standards for HTML coding.  Static files can be pictures and videos, in this project they are mostly javascript backgrounds.
- WSGI templates: index.html, kangaroos.html, ... are aligned with routes in main.py.
- Other templates support WSGI templates.  The base.html template contains common Head, Style, Body, Script definitions.  WSGI templates often "include" or "extend" these templates.  This is a way to reuse code.
- The VANTA javascript statics (backgrounds) are shown and defaulted in base.html (birds), but are block replaced as needed in other templates (solar, net, ...)
- The Bootstrap Navbar code is in navbar.html. The base.html code includes navbar.html.  The WSGI html files extend base.html files.  This is a process of management and correlation to optimize code management.  For instance, if the menu changes discovery of navbar.html is easy, one change reflects on all WSGI html files.
- Jinja2 variables usage is to isolate data and allow redefinitions of attributes in templates.  Observe "{% set variable = %}" syntax for definition and "{{ variable }}" for reference.
- The base.html uses combination of Bootstrap grid styling and custom CSS styling.  Grid styling in observe with the "<Col-3>" markers.  A Bootstrap Grid has a width of 12, thus four "Col-3" markers could fit on a Grid row.
- A key purpose of this project is to embed links to other content.  The "href=" definition embeds hyperlinks into the rendered HTML.  The base.html file shows usage of "href={{github}}", the "{{github}}" is a Jinja2 variable.  Jinja2 variables are pre-processed by Python, a variable swap with value, before being sent to the browser.
