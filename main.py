# The code is importing necessary modules and libraries for building a Flask application.
import zipfile
import librosa
import traceback
import time
import numpy as np
import tensorflow as tf
import io, os, emoji, base64
from flask import Flask, jsonify, request, send_file
from pydub import AudioSegment
from pydub.utils import make_chunks
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)


def process_audio(audio, samplerate=16000):
    """
    The function `process_audio` takes an audio file and converts it into the required format for a
    tflite model to assess by extracting MFCC features and ensuring the shape matches the model input
    shape.
    
    :param audio: The audio parameter is the input audio data that you want to process. It should be in
    a format that can be converted to a numpy array, such as an audio file or an audio stream
    :param samplerate: The samplerate parameter specifies the sample rate of the audio signal. It is
    typically measured in Hz (samples per second). In this case, the default value is set to 16000 Hz,
    defaults to 16000 (optional)
    :return: the processed audio in the required format for the tflite model to assess. The processed
    audio is represented as a numpy array with shape (1, 59, 13), where 1 is the batch dimension, 59 is
    the time steps, and 13 is the number of MFCC features.
    """
    """Convert Audio into required format for tflite model to assess"""
    # convert to numpy array
    snippet = np.array(audio.get_array_of_samples())

    # Ensure the audio is floating-point
    audio = snippet.astype(np.float32) / np.iinfo(np.int16).max

    # Extract MFCC features as per the original Colab code
    mfcc = librosa.feature.mfcc(
        y=audio, sr=samplerate, n_mfcc=13, n_fft=2048, hop_length=512
    )

    # Ensure the shape is consistent with the model input shape
    if mfcc.shape[1] < 59:
        mfcc = np.pad(mfcc, ((0, 0), (0, 59 - mfcc.shape[1])), mode="constant")

    Sxx = mfcc.T.astype(np.float32)
    Sxx = np.expand_dims(Sxx, axis=0)  # Add batch dimension

    # Ensure the shapes match (modify as per your use case)
    if Sxx.shape[1] < 59:
        pad_width = 59 - Sxx.shape[1]
        Sxx = np.pad(Sxx, ((0, 0), (0, pad_width), (0, 0)), mode="constant")

    return Sxx


def create_circumplex_chart(valence, arousal, emotion_label):
    """
    The `create_circumplex_chart` function creates a circumplex chart visualizing emotions based on
    valence and arousal values.
    
    :param valence: Valence represents the emotional positivity or negativity of an emotion. It ranges
    from -1 (negative) to 1 (positive)
    :param arousal: A measure of the intensity or activation level of an emotion. It ranges from low
    arousal (calm, relaxed) to high arousal (excited, agitated)
    :param emotion_label: The emotion label is a string that represents the predicted emotion for the
    given valence and arousal values. It is used to annotate the chart with the predicted emotion label
    :return: a byte array that represents the generated circumplex chart image.
    """
    """Display results as visualisations"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set the background to black
    ax.set_facecolor("black")
    fig.set_facecolor("black")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Differentiate the axis colors
    ax.axhline(0, color="lightblue", linewidth=0.5)
    ax.axvline(0, color="lightcoral", linewidth=0.5)

    # Set labels and title
    ax.set_ylabel("Arousal", color="white", rotation=0, labelpad=50, fontsize=12)
    ax.yaxis.set_label_coords(-0.2, 0.5)
    plt.title(
        "Circumplex Model of Emotion",
        color="white",
        pad=50,
        fontsize=16,
        fontweight="bold",
    )

    # Draw a white circle on the perimeter
    circle = plt.Circle((0, 0), 1.25, color="white", fill=False, linewidth=0.5)
    ax.add_artist(circle)

    # Plot the given emotion with a smaller red dot
    ax.scatter(valence, arousal, s=35, color="red")  # Adjusted size

    # Adjust the position of the label based on valence and arousal
    if valence > 0 and arousal > 0:  # Quadrant I
        ha_val = "left"
        offset_x = 0.05
        offset_y = -0.05
    elif valence > 0:  # Quadrant II
        ha_val = "left"
        offset_x = 0.05
        offset_y = 0.05
    elif arousal > 0:  # Quadrant IV
        ha_val = "right"
        offset_x = -0.05
        offset_y = -0.05
    else:  # Quadrant III
        ha_val = "right"
        offset_x = -0.05
        offset_y = 0.05

    ax.annotate(
        f"Predicted {emotion_label}\n({valence:.2f}, {arousal:.2f})",
        (valence + offset_x, arousal + offset_y),
        fontsize=10,
        ha=ha_val,
        color="white",
    )

    # Add quadrant information
    quadrant_info = {
        "Quadrant I\nHigh Arousal, Positive Valence": (1.25, 1.05),
        "Quadrant II\nLow Arousal, Positive Valence": (1.25, -1.2),
        "Quadrant III\nLow Arousal, Negative Valence": (-1.25, -1.2),
        "Quadrant IV\nHigh Arousal, Negative Valence": (-1.25, 1.05),
    }
    for info, (x, y) in quadrant_info.items():
        ax.text(x, y, info, ha="center", va="center", color="white", fontsize=12)

    # Plot emojis separately
    emojis = {
        emoji.emojize(":grinning_face_with_big_eyes:"): (1.2, 0.9),
        emoji.emojize(":relieved_face:"): (1.2, -1.35),
        emoji.emojize(":pensive_face:"): (-1.2, -1.35),
        emoji.emojize(":angry_face:"): (-1.2, 0.9),
    }
    for emo, (x, y) in emojis.items():
        ax.text(x, y, emo, ha="center", va="center", fontsize=20, color="yellow")

    # Adjust the y-axis to start from the x-axis at 0
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Set the tick parameters
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # Explicitly position the "Valence" label
    ax.text(0, -1.3, "Valence", ha="center", va="center", color="white", fontsize=12)

    # Save the figure with a black background
    byte_arr = io.BytesIO()
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(
        byte_arr, format="png", facecolor="black", bbox_inches="tight", pad_inches=0.5
    )
    plt.close()

    byte_arr.seek(0)

    return byte_arr.getvalue()

@app.route("/", methods=["GET"])
def home():
    # Test API
    return "Hello, World!"

@app.route("/predict", methods=["POST"])
def predict():
    """
    The `predict` function is an API endpoint that takes an audio file, processes it, and returns a zip
    file containing plots of emotion predictions for each slice of the audio.
    :return: The code is returning a zip file containing plots generated from the audio file.
    """
    # Main API for prediction
    try:

        speech = io.BytesIO()
        speech.seek(0)

        # get the audio file from API request message
        # save it to the server
        audio_file = request.files["file"].save(speech)

        speech = AudioSegment.from_wav(speech)

        speech = speech.set_channels(1)

        # Slice speech to eaach second
        speech_slices = make_chunks(speech, 1000)

        plot_imgs = []
        emotion_mapping = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise",
        }

        # Create a new BytesIO object for the zip file
        zip_buffer = io.BytesIO()
        # Create a ZipFile object
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for slice_count, audio in enumerate(speech_slices):
                # Load the TFLite model and allocate tensors.
                interpreter = tf.lite.Interpreter(
                    model_path=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "models",
                        "model4.tflite",
                    )
                )

                interpreter.allocate_tensors()

                # Get input and output tensors.
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                output_details = sorted(output_details, key=lambda x: x["name"])

                # Test the model on sliced audio.
                input_data = process_audio(audio)
                interpreter.set_tensor(input_details[0]["index"], input_data)

                interpreter.invoke()

                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                predictions = [
                    interpreter.get_tensor(output["index"]) for output in output_details
                ]

                valence_score = predictions[0][0][0]
                arousal_score = predictions[2][0][0]
                emotion_label = np.argmax(predictions[1][0])
                encoded_img = create_circumplex_chart(
                    valence_score, arousal_score, emotion_mapping[emotion_label]
                )
                # Write the plot to the ZipFile with a unique name
                zip_file.writestr(f"plot_{slice_count}.png", encoded_img)

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name="plots.zip",
        )
    except:
        return jsonify({"trace": traceback.format_exc()})


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])  # This is for a command-line input  # noqa: F821
    except:
        port = 12345  # If you don't provide any port the port will be set to 12345
    start_time = time.time()
    app.run(debug=False)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    """This API is created to analyse audio and fond the emotions embedded in it.
       To do so, the audio is processed and passed through a tflite model.
       This model provides three outputs, valence, arousal and emotion score.
       Based on these values a visualisation is created for every one second of the speech.
       This is provided as an output. 
    """

