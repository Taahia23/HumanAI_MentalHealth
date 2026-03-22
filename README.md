### Bangla Mental Health Emotion Detection and Human–AI Interaction Prototype

Live link : https://huggingface.co/spaces/taahia26/bangla-mental-health-ai

This project presents a prototype AI system that detects emotional signals from Bangla text and provides supportive responses through an interactive interface. The system explores the integration of **Natural Language Processing** and **Human–Computer Interaction** to support mental health awareness in low-resource languages.

The goal of the project is to demonstrate how AI models can be integrated into a user-facing interface that allows people to interact with emotion detection systems in a simple and accessible way.

### Features
 - Detects emotional signals from Bangla text inputs
 - Supports multiple emotion classes: **happy, neutral, sad, depression**
 - Provides simple supportive suggestions based on predicted emotion
 - Interactive web interface for user input and feedback
 - Deployed as a live demo for testing and research purposes

### Project Architecture
User Input (Bangla Text) &rarr; Text Preprocessing &rarr; TF-IDF Feature Extraction &rarr; Logistic Regression Classifier &rarr; Emotion Prediction &rarr; Supportive &rarr; Suggestion &rarr; Gradio Web Interface

### Model Training
The baseline model was trained using TF-IDF features and a Logistic Regression classifier.
Emotion classes: happy, neutral, sad, depression
### Baseline performance:
- Accuracy: 0.88
- Macro F1 Score: 0.88

### Human–AI Interaction Evaluation
To evaluate the usability of the prototype, a small user study was conducted.
Survey link: https://forms.gle/XSyqv1kP1FiY8DGN8

### Future Improvements
- Larger Bangla mental health dataset
- Explainable AI for emotion predictions
- Larger scale user studies for Human–AI interaction analysis
