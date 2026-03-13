import gradio as gr
import joblib

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_emotion(text):

    vec = vectorizer.transform([text])

    pred = model.predict(vec)[0]

    if pred == "depression":
        message = "You may be feeling distressed. Consider talking to someone you trust."

    elif pred == "sad":
        message = "It seems you are feeling sad."

    elif pred == "happy":
        message = "Glad to see positive emotion."
        
    else:
        message = "Emotion detected."

    return pred, message


interface = gr.Interface(
    fn=predict_emotion,
    inputs="text",
    outputs=["text", "text"],
    title="Bangla Mental Health AI"
)

interface.launch()