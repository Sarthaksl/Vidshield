import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.pipeline_orchestrator import process_pipeline

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'app/static/uploads'
PROCESSED_FOLDER = 'app/static/processed'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    uploader = request.form['uploader']
    file = request.files['video']
    if file and file.filename.endswith('.mp4'):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        # Call your processing pipeline here
        results = process_pipeline(save_path, uploader)

        # Copy output videos to static/processed for web access
        static_processed = app.config['PROCESSED_FOLDER']
        os.makedirs(static_processed, exist_ok=True)

        # Copy tampered video
        tampered_src = results['tampered_video']
        tampered_dest = os.path.join(static_processed, os.path.basename(tampered_src))
        if not os.path.exists(tampered_src):
            return f"Error: Tampered video not found at {tampered_src}", 500
        shutil.copy2(tampered_src, tampered_dest)

        # Copy heatmap video
        heatmap_src = results['heatmap_video']
        heatmap_dest = os.path.join(static_processed, os.path.basename(heatmap_src))
        if not os.path.exists(heatmap_src):
            return f"Error: Heatmap video not found at {heatmap_src}", 500
        shutil.copy2(heatmap_src, heatmap_dest)

        return render_template(
            'results.html',
            original_filename=os.path.basename(results['original_video']),
            tampered_filename=os.path.basename(tampered_dest),
            heatmap_filename=os.path.basename(heatmap_dest),
            status=results['status']
        )
    else:
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
