import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np


UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'flv', 'wmv', 'mov'}

app = Flask(__name__, static_url_path="/static")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 100mb
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


def segmentation(img1, img2):
    # difference of pixel values between the two images
    diff = cv2.absdiff(img1, img2)

#    # If the difference is below diff_threshold, set it to 0
#    height, width, channels = diff.shape
#    DIFF_THRESHOLD = 75
#    for x in range(0, height):
#        for y in range(0, width):
#            if diff[x, y, 0] < DIFF_THRESHOLD and diff[x, y, 1] < DIFF_THRESHOLD \
#                    and diff[x, y, 2] < DIFF_THRESHOLD:
#                diff[x, y, 0] = 0
#                diff[x, y, 1] = 0
#                diff[x, y, 2] = 0

    # Convert diff to gray image
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    threshold = 45
    # Return an array of True-False where True represents value of mask is greater than threshold
    imask = mask > threshold
    # Return an array of zeros with the same shape and type of img2
    segmented_image = np.zeros_like(img2, np.uint8)
    # Mask out image2 using newly created mask
    segmented_image[imask] = img2[imask]

    return segmented_image


def video_processing(original_video, processed_video):
    # Capture frames from video
    VIDEO_SIZE = (1200, 900)
    cap = cv2.VideoCapture(original_video)
    # Write a new video in mp4 format
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Write the video to the download folder
    out = cv2.VideoWriter(app.config['DOWNLOAD_FOLDER'] + processed_video, fourcc, 20, VIDEO_SIZE)
    # First image used as a background to compare with other images
    success, image1 = cap.read()

    while success:
        # Read frame from video
        success, image2 = cap.read()
        if success:
            image = segmentation(image1, image2)
            resized_image = cv2.resize(image, VIDEO_SIZE)

            out.write(resized_image)
            
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def allowed_file(filename):
    # Make sure that the file uploaded has an accepted format
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Redirect to the url if the file uploaded is not accepted or no file was uploaded
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)

        # Process the file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Save original file to upload folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Process the video
            video_processing(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)

            # Automatically download the processed file to the local Downloads folder
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Securely send a file from download folder
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)
