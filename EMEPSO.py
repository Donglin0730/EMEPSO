import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from shapely.ops import unary_union
from scipy.spatial import Voronoi
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import pandas as pd
import time

num_bases = 50  
num_points = 20  

num_particles = 50 
w = 0.5 
c1 = 1.5  
c2 = 2.0  
initial_radius = 5
max_radius = 5  
total_area = 50 * 50  

boundary = box(0, 0, 50, 50)

# Calculation of base station coverage area based on the principle of tolerance exclusion
def calculate_union_area(centers, radii):
    shapely_circles = [Point(x, y).buffer(r).intersection(boundary) for (x, y), r in zip(centers, radii)]
    union_area = unary_union(shapely_circles).area
    return union_area

def fitness_function(particles):
    num_particles = particles.shape[0]
    fitness_values = np.zeros(num_particles)
    
    for i in range(num_particles):
        x = particles[i].reshape(num_bases, 2)
        
        distances = np.array([np.linalg.norm(x - point, axis=1) for point in points])
        
        # Using Voronoi diagram to divide the coverage area of base stations
        vor = Voronoi(x)

        regions, vertices = voronoi_finite_polygons_2d(vor)
        
        # Dynamically adjust the radius of each base station
        for j in range(num_bases):
            region = regions[j]
            polygon = vertices[region]
            polygon = np.clip(polygon, 0, 50)  
            distances_to_polygon = np.array([np.linalg.norm(polygon - point, axis=1) for point in points])
            min_distances = np.min(distances_to_polygon, axis=1)
            R[i][j] = min(np.max(min_distances), initial_radius)
        
        A_total = calculate_union_area(x, R[i])

        signal_coverage = np.zeros(num_points)
        load_distribution = np.zeros(num_bases)
        for j in range(num_bases):
            for k in range(num_points):
                dist = distances[k][j]
                if dist <= R[i][j]:
                    provided_signal = L[j] / (dist**2+1) 
                    actual_signal = min(provided_signal, S[k])
                    signal_coverage[k] += actual_signal
                    load_distribution[j] += actual_signal

        fitness = -A_total / total_area

        shapely_circles = [Point(xy).buffer(r).intersection(boundary) for xy, r in zip(x, R[i])]
        total_circle_area = sum(circle.area for circle in shapely_circles)
        overlap_area = total_circle_area - A_total

        if overlap_area > 0.85 * total_area:
            fitness += 10000 * overlap_area

        for k in range(num_points):
            if signal_coverage[k] < S[k]:
                fitness += 10000 * (S[k] - signal_coverage[k])

        for j in range(num_bases):
            if load_distribution[j] > L[j]:
                fitness += 10000 * (load_distribution[j] - L[j])

        for k in range(num_points):
            if signal_coverage[k] == 0:
                fitness += 10000 

        fitness_values[i] = fitness * 100

    return fitness_values

def voronoi_finite_polygons_2d(vor, radius=None):
    new_regions = []
    new_vertices = vor.vertices.tolist() 
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    all_ridges = {} 
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

# Visualization
def plot_bases_and_points_with_images(bases, radii, points, base_image_path, num, m):
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], color='red', label='Special Points')
    
    base_image = plt.imread(base_image_path)
    for base in bases:
        imagebox = OffsetImage(base_image, zoom=0.065)
        ab = AnnotationBbox(imagebox, base, frameon=False)
        plt.gca().add_artist(ab)
    
    for base, radius in zip(bases, radii):
        circle = plt.Circle(base, radius, color='green', alpha=0.3, fill=True)
        plt.gca().add_artist(circle)
    
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.legend()

import numpy as np
from scipy.sparse import dok_matrix
from sklearn.cluster import KMeans

dim = 2
bounds = np.array([0, 50])
num_iterations = 200
L0 = 5
L_min = 0.1
alpha = 0.01
alpha_RL = 0.01
gamma = 0.6 
epsilon = 0.1
memory_size = 10
num_clusters = 10 
gbest_R = np.full(num_bases, max_radius)

def initialize_particles(num_particles, dim, bounds):
    x = np.random.uniform(bounds[0], bounds[1], (num_particles, num_bases * dim))
    v = np.random.uniform(-1, 1, (num_particles, num_bases * dim))
    return x, v

# Calculate the historical optimal position of the group
def find_global_best(particles, fitness_values):
    best_index = np.argmin(fitness_values)
    return particles[best_index], fitness_values[best_index]

# Calculate the center point of the cube where the particles are located
def calculate_cube_center(particles, L):
    return np.floor(particles / L) * L + L / 2

# Determine whether particles are in the optimal space of the population
def is_in_optimal_space(particles, g_best, L):
    return np.all(np.floor(particles / L) == np.floor(g_best / L), axis=1)

def initialize_Q_table(num_states, num_actions):
    return dok_matrix((num_states, num_actions), dtype=np.float64)

def select_action(Q_table, state, epsilon):
    if (np.random.rand() < epsilon):
        return np.random.randint(Q_table.shape[1])
    else:
        return np.argmax(Q_table[state].toarray())
    
def update_Q_table(Q_table, state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(Q_table[next_state].toarray())
    td_target = reward + gamma * Q_table[next_state, best_next_action]
    td_error = td_target - Q_table[state, action] 
    Q_table[state, action] += alpha * td_error

def calculate_reward(particles, fitness_values):
    return -fitness_values

def get_state(particles, kmeans):
    labels = kmeans.predict(particles)
    return labels

def get_next_state(particles, kmeans):
    return get_state(particles, kmeans)

def generate_adaptive_perturbation(dim, iteration, max_iterations, initial_scale=0.1, final_scale=0.001):
    scale = initial_scale - (initial_scale - final_scale) * (iteration / max_iterations)
    return np.random.uniform(-scale, scale, num_bases * dim)

for m in range(2):
    for n in range(20):
        if m==0:
            points = np.random.rand(num_points, 2) * 50 
        else:
            numbers = [49.4106, 46.7436, 33.9351, 46.5169, 39.8441, 3.3032, 16.2319, 11.7946, 22.63, 28.07, 5.5892, 34.5563, 20.1567, 15.8537, 9.7553, 40.7016, 29.1069, 40.9823, 11.2402, 33.0951, 23.5586, 46.3177, 13.0568, 49.4367, 1.5673, 13.8973, 38.3016, 5.9595, 32.8686, 44.9280, 12.3144, 13.7598, 10.8504, 34.5010, 11.8906, 45.5260, 1.1090, 6.4030, 14.7973, 19.8139]
            number_array = np.array(numbers)
            points = number_array.reshape(-1, 2)

        L = np.random.rand(num_bases) * 100 + 100  
        S = np.random.rand(num_points) * 10 + 10 
        R = np.full((num_particles, num_bases), initial_radius)

        particles, velocities = initialize_particles(num_particles, dim, bounds)
        pbest = particles.copy()
        pbest_fitness = fitness_function(pbest)
        g_best, g_best_fitness = find_global_best(particles, pbest_fitness)

        num_states = num_clusters
        num_actions = 3  # Three actions: learning from the historical best point of the group, learning from the center point of the optimal space of the group, and expanding exploration
        Q_table = initialize_Q_table(num_states, num_actions)
        memory = []

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(particles)

        data = []
        Sum_time = []
        for t in range(num_iterations):
            start_time = time.time() 
            w_t = w * (1 - t / num_iterations)

            # Evolutionary spatial sub-domains
            L_t = L_min + (L0 - L_min) * np.exp(-alpha * t)
            cube_centers = calculate_cube_center(particles, L_t)
            g_cube_center = calculate_cube_center(g_best, L_t)
            in_optimal_space = is_in_optimal_space(particles, g_best, L_t)
            
            states = get_state(particles, kmeans)
            actions = np.array([select_action(Q_table, state, epsilon) for state in states])

            # Reinforcement Learning Driven Particle Navigation Strategy
            perturbations = np.array([generate_adaptive_perturbation(dim, t, num_iterations) for _ in range(num_particles)])
            velocities = np.where(
                actions[:, None] == 0,
                w_t * velocities + c1 * np.random.rand(num_particles, num_bases * dim) * (pbest - particles) + c2 * np.random.rand(num_particles, num_bases * dim) * (g_best - particles) + np.random.rand(num_particles, num_bases * dim) * perturbations,
                np.where(
                    actions[:, None] == 1,
                    w_t * velocities + c1 * np.random.rand(num_particles, num_bases * dim) * (pbest - particles) + c2 * np.random.rand(num_particles, num_bases * dim) * (g_cube_center - particles),
                    w_t * velocities + perturbations
                )
            )

            particles += velocities
            LOGIC = (particles < bounds[0]) | (particles > bounds[1])
            U = np.random.rand(num_particles, num_bases * dim) * (bounds[1] - bounds[0]) + bounds[0]
            particles = LOGIC * U + (1 - LOGIC) * particles
            
            fitness_values = fitness_function(particles)
            
            rewards = calculate_reward(particles, fitness_values)
            next_states = get_next_state(particles, kmeans)
            
            for i in range(num_particles):
                update_Q_table(Q_table, states[i], actions[i], rewards[i], next_states[i], alpha_RL, gamma)
            
                better_fitness_mask = fitness_values < pbest_fitness
                pbest = np.where(better_fitness_mask[:, None], particles, pbest)
                pbest_fitness = np.where(better_fitness_mask, fitness_values, pbest_fitness)
                
                current_g_best, current_g_best_fitness = find_global_best(particles, fitness_values)
                if current_g_best_fitness < g_best_fitness:
                    g_best = current_g_best
                    g_best_fitness = current_g_best_fitness
                    gbest_R = R[np.argmin(pbest_fitness)].copy()
            
            if len(memory) < memory_size:
                memory.append((g_best.copy(), g_best_fitness))
            else:
                worst_index = np.argmax([mem[1] for mem in memory])
                if g_best_fitness < memory[worst_index][1]:
                    memory[worst_index] = (g_best.copy(), g_best_fitness)
            
            # Random restart and historical memory backtracking mechanism
            num_reinit = int(0.1 * num_particles)
            worst_indices = np.argsort(fitness_values)[-num_reinit:]
            particles[worst_indices], velocities[worst_indices] = initialize_particles(num_reinit, dim, bounds)
            
            if memory:
                best_memory = min(memory, key=lambda x: x[1])
                for idx in worst_indices:
                    particles[idx] = best_memory[0] + generate_adaptive_perturbation(dim, t, num_iterations)
            
            end_time = time.time()  
            elapsed_time = end_time - start_time  
            Sum_time.append(elapsed_time)
            print("代码执行时间:", elapsed_time, "秒")
            data.append(g_best_fitness)
            print(f"Iteration {t + 1}: gbest_fitness = {g_best_fitness:.2f}")

        g_best = g_best.reshape(num_bases, 2)
        base_image_path = 'base_station.png'
        plot_bases_and_points_with_images(g_best, gbest_R, points, base_image_path, n, m)
