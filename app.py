from flask import Flask, render_template, Response, request, send_from_directory
from FER_NAIM_EMOTIONDETECTION import detect_emotion
import base64
import threading
from queue import Queue
from threading import Event

app = Flask(__name__, template_folder='templates')

emotion_source = None  
emotion_queue = Queue()
stop_flag = Event()

@app.route("/")
def index():
    return render_template("homepage.html")

@app.route("/homepage")
def homepage():
    return render_template("index.html")

def generate_emotion():
    while True:
        frame_bytes, emotion_label = emotion_queue.get()
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
        yield f'data: {frame_base64} {emotion_label}\n\n'

@app.route('/video_feed')
def video_feed():
    return Response(generate_emotion(), content_type='text/event-stream')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global emotion_source, stop_flag
    if emotion_source is None:
        emotion_source = emotion_queue
        stop_flag = Event()  
        threading.Thread(target=detect_emotion, args=(emotion_source, stop_flag), daemon=True).start()
        return "Emotion detection started"
    else:
        return "Emotion detection already running"

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global emotion_source, stop_flag
    if emotion_source:
        stop_flag.set()
        emotion_source = None
        return "Emotion detection stopped"
    else:
        return "Emotion detection not running"

@app.route('/about')
def about():
    return render_template('about.html')

# Serve images statically
@app.route('/images/<path:path>')
def static_images(path):
    return send_from_directory('images', path)

if __name__ == '__main__':
    app.run(port=9000, debug=True)
