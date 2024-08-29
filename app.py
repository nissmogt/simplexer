import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
import gudhi
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def preprocess_image(image_path, max_points=1000):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")
    
    edges = cv2.Canny(img, 100, 200)
    points = np.column_stack(np.where(edges > 0))
    
    if len(points) == 0:
        raise ValueError("No edges detected in the image")
    
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    return img, points

def gudhi_analysis(point_cloud, max_edge_length, max_dimension=2):
    rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    
    persistence = simplex_tree.persistence()
    betti_numbers = simplex_tree.betti_numbers()
    
    return {
        'betti_numbers': betti_numbers,
        'persistence': persistence,
        'simplex_tree': simplex_tree
    }

def prepare_plot_data(point_cloud, results):
    persistence = results['persistence']
    betti_numbers = results['betti_numbers']
    
    # Prepare data for point cloud scatter plot
    point_cloud_data = point_cloud.tolist()
    
    # Prepare data for persistence barcode
    barcode_data = [{'dimension': p[0], 'start': float(p[1][0]), 'end': float(p[1][1])} for p in persistence]
    
    # Prepare data for persistence diagram
    diagram_data = [{'dimension': p[0], 'birth': float(p[1][0]), 'death': float(p[1][1])} for p in persistence]
    
    # Prepare data for Betti numbers bar chart
    betti_data = [{'dimension': i, 'betti': int(b)} for i, b in enumerate(betti_numbers)]
    
    return {
        'point_cloud': point_cloud_data,
        'barcode': barcode_data,
        'diagram': diagram_data,
        'betti': betti_data
    }

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return redirect(url_for('result', filename=file.filename))
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            return "Image not found", 404

        original_image, point_cloud = preprocess_image(image_path)
        results = gudhi_analysis(point_cloud, max_edge_length=10, max_dimension=2)
        plot_data = prepare_plot_data(point_cloud, results)
        
        # Convert original image to base64 for display
        _, buffer = cv2.imencode('.png', original_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return render_template('result.html', plot_data=plot_data, image_base64=image_base64)
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
