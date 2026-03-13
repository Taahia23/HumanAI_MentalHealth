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


with gr.Blocks() as demo:

    gr.Markdown("# Bangla Mental Health Emotion Detector")
    gr.Markdown("AI system to detect emotions from Bangla text.")

    text_input = gr.Textbox(label="Enter Bangla text")

    emotion_output = gr.Textbox(label="Detected Emotion")
    message_output = gr.Textbox(label="Suggestion")

    btn = gr.Button("Analyze")

    btn.click(
        predict_emotion,
        inputs=text_input,
        outputs=[emotion_output, message_output]
    )

demo.launch()