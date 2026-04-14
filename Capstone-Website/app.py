from flask import Flask
import pickle
from flask import render_template
import ocr
import lineSweep
import normalize
import svm


app = Flask(__name__)



from flask import render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import urllib.request
import os
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads')
app.config['SECRET_KEY'] = 'asldfkjlj'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def home():
    return render_template('home.html')



@app.route('/reload')
def reload_page():
    dir1 = 'static/uploads'
    dir2 = 'static/OCR_Results'
    dir3 = 'static/LineSweep_Results'
    for f in os.listdir(dir1):
        os.remove(os.path.join(dir1, f))

    for f in os.listdir(dir2):
        os.remove(os.path.join(dir2, f))

    for f in os.listdir(dir3):
        os.remove(os.path.join(dir3, f))
    return redirect('/')


@app.route('/process_ocr', methods=['POST'])
def process_image():
    # Clear previous results before each run so files don't accumulate
    for folder in ['static/OCR_Results', 'static/LineSweep_Results']:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except:
                pass

    res = ocr.ocr_algo()
    lineSweep.lineSweep_algo()
    normalize.normalize_all()
    result = svm.svm_algo()

    # flash("Algorithm successfully completed for IFSC Code : " + res)
    if result == "Genuine":
        ret = "Genuine Signature"
        return render_template("home.html", result=ret)
    else:
        ret = "Forged Signature"
        return render_template("home.html", result=ret)

    # return redirect('/')


@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    # print(file)
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded')
        return render_template('home.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

# @app.route("/about")
# def about():
#     return render_template("about.html")


# @app.route("/upload", methods=['GET', 'POST'])
# def upload_image():
#     return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

    '''
    from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import urllib.request
import pickle
import os

import ocr
import lineSweep
import normalize
import svm   


# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads')
app.config['SECRET_KEY'] = 'asldfkjlj'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# ── Model loading ─────────────────────────────────────────────────────────────
# model_auth.pkl is created by svm.py on first run and reused on every
# subsequent request — no retraining overhead per request.
# svm.py already calls _load_model() at import time (above), so the model
# is ready before the first request arrives.
# If you ever need to force a retrain (e.g. after adding new training data):
#   from svm import force_retrain; force_retrain()
# ─────────────────────────────────────────────────────────────────────────────


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/reload')
def reload_page():
    for folder in ['static/uploads', 'static/OCR_Results', 'static/LineSweep_Results']:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                pass
    return redirect('/')


@app.route('/process_ocr', methods=['POST'])
def process_image():
    # Clear previous results so files don't accumulate across requests
    for folder in ['static/OCR_Results', 'static/LineSweep_Results']:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                pass

    # Pipeline: OCR → Line Sweep → Normalize → SVM
    res    = ocr.ocr_algo()
    lineSweep.lineSweep_algo()
    normalize.normalize_all()

    # svm_algo() loads from model_auth.pkl — no retraining on every request
    result = svm.svm_algo()

    if result == "Genuine":
        return render_template("home.html", result="Genuine Signature")
    else:
        return render_template("home.html", result="Forged Signature")


@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        flash('Image successfully uploaded')
        return render_template('home.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
    '''