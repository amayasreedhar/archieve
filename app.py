from flask import *
import os
import numpy as np
import cv2

app=Flask(__name__,template_folder='template')

# path to the prototxt file
prototxt = os.path.join(os.getcwd(), 'detect_faces/deploy.prototxt.txt')
# path to the model file
model = os.path.join(os.getcwd(), 'detect_faces/res10_300x300_ssd_iter_140000.caffemodel')

# importing the face detection architecture from opencv
net = cv2.dnn.readNetFromCaffe(prototxt, model)

#route / for the flask app
@app.route('/')
def upload():
    return render_template("file_upload_form.html")

#route path for the result
@app.route('/result', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        #get the confidence value of the image
        confidence = detect(f)
        #validation
        if confidence > 0.5:
                message = 'Face Detected and File Uploaded'
                #save the file if the confidence values satisfies our condition
                f.save(f.filename)
                return render_template("result.html", name = f.filename, message = message)
        else:
                return render_template("result.html", name = f.filename, message = 'No Face Detectd, File Not saved')


def detect(image):
    # getting the user inputs
    image_bytes = image.read()
    # converting the bytes array to image
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    # extracting height and width from the image
    (h, w) = image.shape[:2]
    # setting the input for the dnn from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    # detecting the face
    detections = net.forward()
    #getting the confidence of the detection
    confidence = detections[0, 0, 0, 2]
    #returns the confidence value
    return confidence


if __name__ == '__main__':
    app.run(debug = True)
