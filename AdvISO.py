import cv2
import numpy as np
import random
from tqdm import tqdm
from scipy.special import comb
from scipy.interpolate import splprep, splev
import os

omega = 0.9
c1 = 1.6
c2 = 1.4
r1 = 0.5
r2 = 0.5


current_pso_image_count = 0
successful_attack_count = 0
detector_query_count = 0 


detector_queries_per_pso_image = []

def initialization(population_size, X1, Y1, X2, Y2, ratio):

    population = np.zeros((population_size, 18))

    box_width = X2 - X1
    box_height = Y2 - Y1

    central_point_constrain_center_x = (X1 + X2) / 2
    central_point_constrain_center_y = Y1 + (box_height * 3/8)  

    central_point_constrain_side = box_width / 2

    central_point_x_min = max(X1, central_point_constrain_center_x - central_point_constrain_side / 2)
    central_point_x_max = min(X2, central_point_constrain_center_x + central_point_constrain_side / 2)
    central_point_y_min = max(Y1, central_point_constrain_center_y - central_point_constrain_side / 2)
    central_point_y_max = min(Y2, central_point_constrain_center_y + central_point_constrain_side / 2)
    central_point_y_max = min(central_point_y_max, Y1 + box_height / 2) 

    for i in range(population_size):
        particle_data = []

        initial_center_x = random.uniform(central_point_x_min, central_point_x_max)
        initial_center_y = random.uniform(central_point_y_min, central_point_y_max)
        particle_data.append(initial_center_x)
        particle_data.append(initial_center_y)

        pattern_width = box_width / ratio
        pattern_height = pattern_width * 2 

        sub_grid_w = pattern_width / 3
        sub_grid_h = pattern_height / 3

        relative_grid_centers = [
            (-sub_grid_w, -sub_grid_h),    
            (0, -sub_grid_h),              
            (sub_grid_w, -sub_grid_h),   
            (sub_grid_w, 0),              
            (sub_grid_w, sub_grid_h),      
            (0, sub_grid_h),             
            (-sub_grid_w, sub_grid_h),     
            (-sub_grid_w, 0)              
        ]

        jitter_x = sub_grid_w / 2
        jitter_y = sub_grid_h / 2

        for j in range(8):
            rel_cx, rel_cy = relative_grid_centers[j]

            point_x_min = initial_center_x + rel_cx - jitter_x
            point_x_max = initial_center_x + rel_cx + jitter_x
            point_y_min = initial_center_y + rel_cy - jitter_y
            point_y_max = initial_center_y + rel_cy + jitter_y

            point_x_min = max(point_x_min, X1)
            point_x_max = min(point_x_max, X2)
            point_y_min = max(point_y_min, Y1)
            point_y_max = min(point_y_max, Y2)

            subgrid_x = random.uniform(point_x_min, point_x_max)
            subgrid_y = random.uniform(point_y_min, point_y_max)

            particle_data.append(subgrid_x)
            particle_data.append(subgrid_y)

        population[i] = np.array(particle_data)

    return population

def clip(particle_position, box, ratio):
    X1, Y1, X2, Y2 = box
    box_width = X2 - X1
    box_height = Y2 - Y1

    central_point_constrain_center_x = (X1 + X2) / 2
    central_point_constrain_center_y = Y1 + (box_height * 3/8) 

    central_point_constrain_side = box_width / 2

    central_point_x_min = max(X1, central_point_constrain_center_x - central_point_constrain_side / 2)
    central_point_x_max = min(X2, central_point_constrain_center_x + central_point_constrain_side / 2)
    central_point_y_min = max(Y1, central_point_constrain_center_y - central_point_constrain_side / 2)
    central_point_y_max = min(Y2, central_point_constrain_center_y + central_point_constrain_side / 2)
    central_point_y_max = min(central_point_y_max, Y1 + box_height / 2) 

    particle_position[0] = np.clip(particle_position[0], central_point_x_min, central_point_x_max)
    particle_position[1] = np.clip(particle_position[1], central_point_y_min, central_point_y_max)

    current_center_x = particle_position[0]
    current_center_y = particle_position[1]

    pattern_width = box_width / ratio
    pattern_height = pattern_width * 2

    sub_grid_w = pattern_width / 3
    sub_grid_h = pattern_height / 3

    relative_grid_centers = [
        (-sub_grid_w, -sub_grid_h),     
        (0, -sub_grid_h),             
        (sub_grid_w, -sub_grid_h),     
        (sub_grid_w, 0),              
        (sub_grid_w, sub_grid_h),       
        (0, sub_grid_h),            
        (-sub_grid_w, sub_grid_h),     
        (-sub_grid_w, 0)              
    ]

    jitter_x = sub_grid_w / 2
    jitter_y = sub_grid_h / 2

    for j in range(8):
        px_idx = 2 + j * 2
        py_idx = 2 + j * 2 + 1 

        rel_cx, rel_cy = relative_grid_centers[j]


        point_x_min = current_center_x + rel_cx - jitter_x
        point_x_max = current_center_x + rel_cx + jitter_x
        point_y_min = current_center_y + rel_cy - jitter_y
        point_y_max = current_center_y + rel_cy + jitter_y

        particle_position[px_idx] = np.clip(particle_position[px_idx], max(point_x_min, X1), min(point_x_max, X2))
        particle_position[py_idx] = np.clip(particle_position[py_idx], max(point_y_min, Y1), min(point_y_max, Y2))

    return particle_position


def create_curve_for_population(population_array):
    all_curve_points = []

    for particle in population_array:

        selected_points = []
        for i in range(2, len(particle), 2): 
            selected_points.append([particle[i], particle[i + 1]])

        if not selected_points:
            all_curve_points.append(None)
            continue

        selected_points.append(selected_points[0]) 

        points_arr = np.array(selected_points)

        epsilon = 1e-6
        filtered_points = [points_arr[0]]
        for k in range(1, len(points_arr)):
            if np.linalg.norm(points_arr[k] - points_arr[k-1]) > epsilon:
                filtered_points.append(points_arr[k])

        points_to_interpolate = np.array(filtered_points)

        min_points_for_spline = 4

        if len(points_to_interpolate) >= min_points_for_spline:
            try:
                if np.allclose(points_to_interpolate[:, 0], points_to_interpolate[0, 0]) or \
                   np.allclose(points_to_interpolate[:, 1], points_to_interpolate[0, 1]):
                    curve_points = points_to_interpolate.reshape((-1, 1, 2)).astype(np.int32)
                    all_curve_points.append(curve_points)
                else:
                    tck, u = splprep(points_to_interpolate.T, s=0, per=1)
                    u_new = np.linspace(0, 1, 100)
                    x_new, y_new = splev(u_new, tck)
                    curve_points = np.array([x_new, y_new]).T.reshape((-1, 1, 2)).astype(np.int32)
                    all_curve_points.append(curve_points)
            except Exception as e:
                print(f"Warning: splprep failed for a particle ({e}). Falling back to polygon.")
                if len(points_to_interpolate) >= 2:
                    curve_points = points_to_interpolate.reshape((-1, 1, 2)).astype(np.int32)
                    all_curve_points.append(curve_points)
                else:
                    all_curve_points.append(None)
        elif len(points_to_interpolate) >= 2:
            curve_points = points_to_interpolate.reshape((-1, 1, 2)).astype(np.int32)
            all_curve_points.append(curve_points)
        else:
            all_curve_points.append(None)

    return all_curve_points

def fitness_function(image, particle_position_list, detect_function, step_num, particle_idx, output_visualization_dir):
    global detector_query_count 

    particle_position = particle_position_list[0]
    result = image.copy()

    single_particle_curves = create_curve_for_population(np.array([particle_position]))

    for curve_points in single_particle_curves:
        if curve_points is not None and len(curve_points) > 0:
            cv2.fillPoly(result, [curve_points], (0,0,0))

    
    os.makedirs(output_visualization_dir, exist_ok=True)
    viz_filename = "process_image.jpg"
    viz_filepath = os.path.join(output_visualization_dir, viz_filename)
    try:
        cv2.imwrite(viz_filepath, result)
    except Exception as e:
        print(f"Error saving visualization image to {viz_filepath}: {e}")

    detector_query_count += 1
    person_boxes = detect_function(result)

    all_detections = []

    if isinstance(person_boxes, tuple) and len(person_boxes) == 2:
        for class_bboxes in person_boxes[0]:
            if class_bboxes.size != 0:
                all_detections.extend(class_bboxes.tolist())
    elif isinstance(person_boxes, list):
        for class_bboxes in person_boxes:
            if class_bboxes.size != 0:
                all_detections.extend(class_bboxes.tolist())

    confidences = []

    for detected_box in all_detections:
        if len(detected_box) >= 5:
            confidences.append(detected_box[4])

    if confidences:
        fitness_score = np.mean(confidences)
    else:
        fitness_score = 0.0 

    print(f"Pic Number: {current_pso_image_count}, Count: {successful_attack_count}, "
          f"Query (current image): {detector_query_count}, PSO Step: {step_num}, Particle Index: {particle_idx}, Fitness: {fitness_score:.4f}")

    return fitness_score

def pso_optimization(image, box, detect_function, output_visualization_dir, population_size=50, max_steps=10, omega=0.9, c1=1.6, c2=1.4):
    X1, Y1, X2, Y2 = box

    population = initialization(population_size, X1, Y1, X2, Y2, ratio=2.0)

    velocities = np.zeros_like(population)
    P_best = np.copy(population)
    P_best_fitness = np.full(population_size, np.inf)

    G_best = np.copy(population[0])
    G_best_fitness = np.inf

    early_stop_triggered = False

    fitness_values = np.zeros(population_size)

    for step_num in tqdm(range(max_steps), desc="PSO Optimization"):
        for i in range(population_size):
            
            fitness_values[i] = fitness_function(image, [population[i]], detect_function, step_num, i, output_visualization_dir)

            if fitness_values[i] < P_best_fitness[i]:
                P_best[i] = population[i]
                P_best_fitness[i] = fitness_values[i]
            
            if fitness_values[i] == 0.0:
                G_best = population[i] 
                G_best_fitness = 0.0
                early_stop_triggered = True
                print(f"\nOptimal fitness of 0.0 achieved by particle {i} at PSO step {step_num}. Stopping optimization early for this image.")
                break 
        
        if early_stop_triggered:
            break 

        if not early_stop_triggered:
            best_particle_idx_in_step = np.argmin(fitness_values)
            if fitness_values[best_particle_idx_in_step] < G_best_fitness:
                G_best = population[best_particle_idx_in_step]
                G_best_fitness = fitness_values[best_particle_idx_in_step]

        for i in range(population_size):
            r1 = 0.5
            r2 = 0.5
            velocities[i] = (omega * velocities[i] +
                             c1 * r1 * (P_best[i] - population[i]) +
                             c2 * r2 * (G_best - population[i]))

            population[i] = population[i] + velocities[i]

            population[i] = clip(population[i], box, ratio=2.0)

    return G_best

def process_image_with_pso(image_path, output_dir, detect_function, metrics):
    global current_pso_image_count, successful_attack_count, detector_query_count, detector_queries_per_pso_image

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load the image from the path {image_path}.")
        return

    metrics['count_all'] += 1

    detector_query_count = 0

    detector_query_count += 1
    person_boxes_raw = detect_function(image) 

    all_initial_detections = []
    if isinstance(person_boxes_raw, tuple) and len(person_boxes_raw) == 2:
        for class_bboxes in person_boxes_raw[0]:
            if class_bboxes.size != 0:
                all_initial_detections.extend(class_bboxes.tolist())
    elif isinstance(person_boxes_raw, list):
        for class_bboxes in person_boxes_raw:
            if class_bboxes.size != 0:
                all_initial_detections.extend(class_bboxes.tolist())
    else:
        print(f"Unexpected raw detection output format: {type(person_boxes_raw)}. Expected list or tuple.")
        return

    if len(all_initial_detections) != 1:
        print(f"Skip image {image_path}: multiple bounding boxes detected or none detected.")
        return

    current_pso_image_count += 1
    metrics['Query_eligible_images'] += 1 

    all_initial_detections.sort(key=lambda x: x[4], reverse=True)
    detection_box = all_initial_detections[0][:4]

    detection_box = [int(x) for x in detection_box]

    print(f"Initial bounding box: {detection_box}")
    if len(detection_box) != 4:
        print(f"Invalid initial bounding box format: {detection_box}. Expected to contain 4 coordinates.")
        return

    current_image_viz_dir = output_dir
    os.makedirs(current_image_viz_dir, exist_ok=True)
    print(f"Visualization image 'process_image.jpg' will be saved to: {current_image_viz_dir}")


    optimized_points = pso_optimization(image, detection_box, detect_function,
                                         current_image_viz_dir, 
                                         population_size=50, max_steps=10, 
                                         omega=0.9, c1=1.6, c2=1.4)

    all_curve_points = create_curve_for_population(np.array([optimized_points]))

    result_image = image.copy()
    for curve_points in all_curve_points:
        if curve_points is not None and len(curve_points) > 0:
            cv2.fillPoly(result_image, [curve_points], (0,0,0))  
        else:
            print("Warning: Skip empty or invalid curve_points.")

    image_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"result_{image_filename}")

    try:
        cv2.imwrite(output_path, result_image)
        print(f"Saved the processed image to {output_path}.")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")

    print("Checking for detections on the modified image (final query)...")
    final_detections_raw = detect_function(result_image)
    detector_query_count += 1 

    final_detections_count = 0
    if isinstance(final_detections_raw, tuple) and len(final_detections_raw) == 2:
        for class_bboxes in final_detections_raw[0]:
            if class_bboxes.size != 0:
                final_detections_count += len(class_bboxes)
    elif isinstance(final_detections_raw, list):
        for class_bboxes in final_detections_raw:
            if class_bboxes.size != 0:
                final_detections_count += len(class_bboxes)

    if final_detections_count == 0:
        print(f"Success! No objects were detected in the processed image at {output_path}.")
        metrics['ASR'] += 1
        successful_attack_count = metrics['ASR'] 
    else:
        print(f"Failure: {final_detections_count} targets detected in the processed image {output_path}.")
    

    detector_queries_per_pso_image.append(detector_query_count)


def main_processing_pipeline(input_image_dir, output_root_dir, detect_function):
    global current_pso_image_count, successful_attack_count, detector_query_count, detector_queries_per_pso_image

    current_pso_image_count = 0
    successful_attack_count = 0
    detector_query_count = 0 
    detector_queries_per_pso_image = [] 

    os.makedirs(output_root_dir, exist_ok=True)

    metrics = {
        'count_all': 0, 
        'Query_eligible_images': 0, 
        'ASR': 0 
    }

    image_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    print(f"Found {len(image_files)} images in {input_image_dir}")

    for image_filename in image_files:
        image_path = os.path.join(input_image_dir, image_filename)
        print(f"\n--- Processing {image_path} ---")
        process_image_with_pso(image_path, output_root_dir, detect_function, metrics)
        print(f"Current Cumulative Metrics: Total Images: {metrics['count_all']}, "
              f"PSO Eligible: {metrics['Query_eligible_images']}, "
              f"Successful Attacks: {metrics['ASR']}")


    print("\n--- Final Results ---")
    print(f"Total images processed (count_all): {metrics['count_all']}")
    print(f"Images with single initial detection (Pic Number concept): {metrics['Query_eligible_images']}")
    
    if metrics['Query_eligible_images'] > 0:
        total_queries_for_pso_images = sum(detector_queries_per_pso_image)
        average_queries_per_pso_image = total_queries_for_pso_images / metrics['Query_eligible_images']
        print(f"Total Detector Queries for PSO-eligible images: {total_queries_for_pso_images}")
        print(f"Average Detector Queries per PSO-eligible image (Query): {average_queries_per_pso_image:.2f}")
        print(f"ASR Rate (Successful Attacks / PSO Eligible Images): {metrics['ASR'] / metrics['Query_eligible_images']:.4f}")
    else:
        print("Average Detector Queries: N/A (No images had a single initial detection to run PSO on)")
        print("ASR Rate: N/A (No images had a single initial detection to run PSO on)")
    
    print(f"Images successfully attacked (Count): {metrics['ASR']}")


# Example Usage:
if __name__ == "__main__":
    try:
        from detect_single_image import yolov3_inf
    except ImportError:
        print("Error: Could not import 'yolov3_inf' from 'detect_single_image.py'.")
        print("Please ensure 'detect_single_image.py' is in the same directory and contains 'yolov3_inf' function.")
        def yolov3_inf(image):
            dummy_bbox = np.array([[200, 200, 400, 400, 0.9, 0]])
            return [dummy_bbox]

    input_directory = '/root/autodl-tmp/.autodl/adv/555'
    output_directory = '/root/autodl-tmp/.autodl/adv/output'

    if not os.path.exists(input_directory):
        os.makedirs(input_directory)
        dummy_image_path = os.path.join(input_directory, "dummy_image.jpg")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(dummy_img, (200, 200), (400, 400), (0, 255, 0), 2)
        cv2.putText(dummy_img, "Dummy Image (Person Detected)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(dummy_image_path, dummy_img)
        print(f"Created a dummy image at {dummy_image_path} for testing.")
        print("Please replace this with your actual image dataset.")

    main_processing_pipeline(input_directory, output_directory, yolov3_inf)