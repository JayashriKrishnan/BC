from flask import Flask, render_template, flash, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os

import ocr
import lineSweep
import normalize
import svm
import database
import signet

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asldfkjlj'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static/uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
DB_SIGNATURES_DIR  = 'static/db_signatures'

# ── Startup: init DB and warm up models ───────────────────────────────────────
database.init_db()
signet.load_model()   # loads signet_model.keras once so first request is fast
# svm._load_model() already runs at import time inside svm.py


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _clear_folders(*folders):
    for folder in folders:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                pass


def _combine_results(signet_verdict, svm_result, distance, user_name, unique_id):
    """
    Build the final combined verdict dict shown in the UI.

    Signet is the main decider (did the signature match the enrolled one?).
    SVM is the qualifier (were the strokes natural?).
    """
    name_str = f"{user_name} ({unique_id})" if user_name else unique_id

    if signet_verdict == "Match":
        if svm_result == "Genuine":
            label   = "APPROVED"
            detail  = f"Signature verified for {name_str}. Strokes appear genuine."
            css     = "success"
        else:
            label   = "SUSPICIOUS"
            detail  = (f"Signature matches records for {name_str}, "
                       "but stroke pattern appears unnatural. Manual review recommended.")
            css     = "warning"
    else:
        if svm_result == "Genuine":
            label   = "REJECTED"
            detail  = (f"Signature does NOT match account records for {name_str}. "
                       "Natural strokes but wrong person.")
            css     = "danger"
        else:
            label   = "REJECTED"
            detail  = (f"Signature does NOT match account records for {name_str} "
                       "and stroke pattern is unnatural.")
            css     = "danger"

    return {
        "label":           label,
        "detail":          detail,
        "css":             css,
        "signet_verdict":  signet_verdict,
        "signet_distance": f"{distance:.4f}",
        "svm_result":      svm_result,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('template.html')


@app.route('/reload')
def reload_page():
    _clear_folders('static/uploads', 'static/OCR_Results', 'static/LineSweep_Results')
    session.pop('unique_id', None)
    return redirect('/')


# ── Step 1: Upload cheque image + capture unique ID ───────────────────────────

@app.route('/predict', methods=['POST'])
def upload_image():
    unique_id = request.form.get('unique_id', '').strip()

    if not unique_id:
        flash('Please enter the account / unique ID.')
        return redirect('/')

    if 'file' not in request.files or request.files['file'].filename == '':
        flash('No cheque image selected.')
        return redirect('/')

    file = request.files['file']
    if not allowed_file(file.filename):
        flash('Allowed image types: png, jpg, jpeg, gif')
        return redirect('/')

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Store the ID in the session so /process_ocr can read it
    session['unique_id']      = unique_id
    session['upload_filename'] = filename

    flash(f'Cheque uploaded successfully. ID: {unique_id}')
    return render_template('template.html', uploaded=True, unique_id=unique_id)


# ── Step 2: Run full pipeline + Signet ────────────────────────────────────────

@app.route('/process_ocr', methods=['POST'])
def process_image():
    unique_id = session.get('unique_id', '').strip()

    if not unique_id:
        flash('No account ID found. Please upload the cheque again.')
        return redirect('/')

    # ── DB lookup ──────────────────────────────────────────────────────────────
    user = database.get_user(unique_id)
    if user is None:
        return render_template(
            'template.html',
            result={
                "label":  "NOT FOUND",
                "detail": f"No account found for ID: '{unique_id}'. "
                          "Please enroll this account first.",
                "css":    "secondary",
                "signet_verdict":  "—",
                "signet_distance": "—",
                "svm_result":      "—",
            }
        )

    genuine_path = user['signature_path']
    user_name    = user['name']

    if not os.path.exists(genuine_path):
        return render_template(
            'template.html',
            result={
                "label":  "ERROR",
                "detail": f"Genuine signature image not found on disk for ID '{unique_id}'. "
                          "Please re-enroll this account.",
                "css":    "danger",
                "signet_verdict":  "—",
                "signet_distance": "—",
                "svm_result":      "—",
            }
        )

    # ── Pipeline: OCR → LineSweep → Normalize ─────────────────────────────────
    _clear_folders('static/OCR_Results', 'static/LineSweep_Results')

    ocr.ocr_algo()
    lineSweep.lineSweep_algo()
    normalize.normalize_all()

    # ── Check pipeline produced output ────────────────────────────────────────
    ls_files = [
        f for f in os.listdir('static/LineSweep_Results')
        if not f.startswith('.')
    ]
    if not ls_files:
        return render_template(
            'template.html',
            result={
                "label":  "EXTRACTION FAILED",
                "detail": "Could not locate a signature region on the cheque. "
                          "Ensure the cheque is clear and contains the words 'please' and 'above'.",
                "css":    "warning",
                "signet_verdict":  "—",
                "signet_distance": "—",
                "svm_result":      "—",
            }
        )

    # ── SVM: naturalness check ─────────────────────────────────────────────────
    svm_result = svm.svm_algo()   # "Genuine" | "Forged"

    # ── Signet: identity match ─────────────────────────────────────────────────
    latest_crop = max(
        [os.path.join('static/LineSweep_Results', f) for f in ls_files],
        key=os.path.getmtime,
    )

    try:
        distance, signet_verdict = signet.compare_signatures(latest_crop, genuine_path)
    except Exception as e:
        print(f"[Signet] Error: {e}")
        return render_template(
            'template.html',
            result={
                "label":  "ERROR",
                "detail": f"Signet model error: {e}",
                "css":    "danger",
                "signet_verdict":  "Error",
                "signet_distance": "—",
                "svm_result":      svm_result,
            }
        )

    # ── Combine and return ─────────────────────────────────────────────────────
    result = _combine_results(signet_verdict, svm_result, distance, user_name, unique_id)
    return render_template('template.html', result=result)


# ── Enrollment: add a user + their genuine signature to the DB ─────────────────

@app.route('/enroll', methods=['GET'])
def enroll_form():
    return render_template('enroll.html')


@app.route('/enroll', methods=['POST'])
def enroll_submit():
    unique_id = request.form.get('unique_id', '').strip()
    name      = request.form.get('name', '').strip()

    if not unique_id:
        flash('Account ID is required.')
        return redirect('/enroll')

    if 'signature' not in request.files or request.files['signature'].filename == '':
        flash('Please upload the genuine signature image.')
        return redirect('/enroll')

    sig_file = request.files['signature']
    if not allowed_file(sig_file.filename):
        flash('Allowed image types: png, jpg, jpeg, gif')
        return redirect('/enroll')

    os.makedirs(DB_SIGNATURES_DIR, exist_ok=True)
    ext      = sig_file.filename.rsplit('.', 1)[1].lower()
    sig_name = f"{secure_filename(unique_id)}.{ext}"
    sig_path = os.path.join(DB_SIGNATURES_DIR, sig_name)
    sig_file.save(sig_path)

    database.add_user(unique_id, name, sig_path)

    flash(f"User '{unique_id}' enrolled successfully.")
    return redirect('/enroll')


if __name__ == '__main__':
    app.run(debug=True)
