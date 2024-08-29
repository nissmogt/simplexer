import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import gudhi

def preprocess_image(image_path, max_points=1000):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    points = np.column_stack(np.where(edges > 0))

    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    return img, points

def gudhi_analysis(point_cloud, max_edge_length, max_dimension=2):
    start_time = time.time()
    start_cpu = psutil.cpu_percent()
    
    rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    
    persistence = simplex_tree.persistence()
    betti_numbers = simplex_tree.betti_numbers()
    
    end_time = time.time()
    end_cpu = psutil.cpu_percent()
    
    return {
        'time': end_time - start_time,
        'cpu': (start_cpu + end_cpu) / 2,
        'betti_numbers': betti_numbers,
        'persistence': persistence,
        'simplex_tree': simplex_tree
    }

def plot_results(original_image, point_cloud, results):
    fig = plt.figure(figsize=(20, 15))
    
    # Original Image
    ax1 = fig.add_subplot(231)
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    
    # Point Cloud
    ax2 = fig.add_subplot(232)
    ax2.scatter(point_cloud[:, 1], point_cloud[:, 0], s=1)
    ax2.set_title('Point Cloud')
    ax2.invert_yaxis()
    
    # Persistence Barcode
    ax3 = fig.add_subplot(233)
    gudhi.plot_persistence_barcode(results['persistence'], axes=ax3)
    ax3.set_title('Persistence Barcode')
    
    # Persistence Diagram
    ax4 = fig.add_subplot(234)
    gudhi.plot_persistence_diagram(results['persistence'], axes=ax4)
    ax4.set_title('Persistence Diagram')
    
    # Betti Numbers
    ax5 = fig.add_subplot(235)
    dimensions = range(len(results['betti_numbers']))
    ax5.bar(dimensions, results['betti_numbers'])
    ax5.set_title('Betti Numbers')
    ax5.set_xlabel('Dimension')
    ax5.set_ylabel('Betti Number')
    
    # Performance Metrics
    ax6 = fig.add_subplot(236)
    metrics = ['Time (s)', 'CPU Usage (%)']
    values = [results['time'], results['cpu']]
    ax6.bar(metrics, values)
    ax6.set_title('Performance Metrics')
    
    plt.tight_layout()
    plt.show()

def main(image_path, max_edge_length, max_dimension=2, max_points=1000):
    original_image, point_cloud = preprocess_image(image_path, max_points)
    results = gudhi_analysis(point_cloud, max_edge_length, max_dimension)
    
    print("GUDHI Analysis Results:")
    print(f"Time: {results['time']:.2f} seconds")
    print(f"CPU Usage: {results['cpu']:.2f}%")
    print(f"Betti Numbers: {results['betti_numbers']}")
    
    plot_results(original_image, point_cloud, results)
    
    return results
# Usage
image_file = './office_chair.jpg'
results = main(image_file, max_edge_length=10, max_dimension=2, max_points=1000)


