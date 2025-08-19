import os
import io
import gpxpy
import gpxpy.gpx
import json
from flask import Flask, render_template, request, url_for, session

# Importar matplotlib y configurar el backend 'Agg' antes que cualquier otro import de matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# App configuration
app = Flask(__name__)
# A secret key is required for Flask sessions
app.secret_key = 'super_secret_key_for_session'
# Define the folder for static files (where plots will be saved)
app.config['STATIC_FOLDER'] = os.path.join(app.root_path, 'static', 'plots')
# Ensure the plots folder exists
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def generate_plot(gpx_data, pa_locations, plot_type='altimetry'):
    """
    Generates an altimetry plot or a 2D map and returns it as a byte buffer.
    """
    if not gpx_data.tracks or not gpx_data.tracks[0].segments:
        raise ValueError("El archivo GPX no contiene datos de tracks o segmentos.")
        
    points = gpx_data.tracks[0].segments[0].points
    
    # Extract data from points
    lats = [p.latitude for p in points]
    lons = [p.longitude for p in points]
    alts = [p.elevation for p in points]
    
    # Calculate the cumulative distance of the points
    dists = [0]
    for i in range(1, len(points)):
        d = points[i].distance_2d(points[i-1])
        dists.append(dists[-1] + d)
    
    dists_km = [d / 1000 for d in dists]
    
    # Configure plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pa_coords = []
    for pa_dist_km in pa_locations:
        closest_point_index = np.argmin(np.abs(np.array(dists_km) - pa_dist_km))
        closest_point = points[closest_point_index]
        pa_coords.append((closest_point.longitude, closest_point.latitude, closest_point.elevation))

    if plot_type == 'altimetry':
        ax.plot(dists_km, alts, color='#1f77b4', linewidth=2, label='Perfil de Altimetría')
        ax.set_xlabel('Distancia (km)', fontsize=12)
        ax.set_ylabel('Altitud (m)', fontsize=12)
        ax.set_title('Perfil de Altimetría del Recorrido', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for i, pa in enumerate(pa_coords):
            pa_x = dists_km[np.argmin(np.abs(np.array(dists_km) - pa_locations[i]))]
            pa_y = pa[2]
            ax.plot(pa_x, pa_y, 'o', color='#ff7f0e', markersize=10, label=f'PA {i+1}' if i==0 else "")
            ax.text(pa_x, pa_y + 10, f' PA {i+1}', verticalalignment='bottom', fontsize=10)

    elif plot_type == 'map':
        ax.plot(lons, lats, color='#2ca02c', linewidth=2, label='Recorrido del Circuito')
        ax.set_xlabel('Longitud', fontsize=12)
        ax.set_ylabel('Latitud', fontsize=12)
        ax.set_title('Plano 2D del Circuito', fontsize=16, fontweight='bold')
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for i, pa in enumerate(pa_coords):
            ax.plot(pa[0], pa[1], 'o', color='#ff7f0e', markersize=10, label=f'PA {i+1}' if i==0 else "")
            ax.text(pa[0], pa[1], f' PA {i+1}', verticalalignment='bottom', fontsize=10)
    
    ax.legend()
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def analyze_segments(gpx_data, pa_locations, sub_segment_length):
    """
    Analyzes and calculates metrics for sub-segments between PAs.
    
    Args:
        gpx_data (gpxpy.gpx.GPX): Parsed GPX object.
        pa_locations (list): List of PA locations in KM.
        sub_segment_length (int): Length of each sub-segment in meters.
        
    Returns:
        list: A list of dictionaries with metrics for each sub-segment.
    """
    if not gpx_data.tracks or not gpx_data.tracks[0].segments:
        return []

    points = gpx_data.tracks[0].segments[0].points
    
    dists = [0]
    for i in range(1, len(points)):
        d = points[i].distance_2d(points[i-1])
        dists.append(dists[-1] + d)
    
    pa_locations_m = [loc * 1000 for loc in pa_locations]
    pa_locations_m.insert(0, 0)
    pa_locations_m.append(dists[-1])
    
    results = []
    cumulative_gain = 0
    cumulative_loss = 0
    
    # Iterate through each segment between PAs (or start/end of the route)
    for i in range(len(pa_locations_m) - 1):
        start_dist_pa = pa_locations_m[i]
        end_dist_pa = pa_locations_m[i+1]
        
        # Find the point indices for the PA-to-PA segment
        start_index_pa = np.argmin(np.abs(np.array(dists) - start_dist_pa))
        end_index_pa = np.argmin(np.abs(np.array(dists) - end_dist_pa))

        # Generate sub-segments
        current_sub_dist_from_pa_start = 0
        while current_sub_dist_from_pa_start < (end_dist_pa - start_dist_pa):
            segment_start_dist = start_dist_pa + current_sub_dist_from_pa_start
            
            segment_end_dist = min(start_dist_pa + current_sub_dist_from_pa_start + sub_segment_length, end_dist_pa)

            # Find the GPX points corresponding to the sub-segment
            start_point_index = np.argmin(np.abs(np.array(dists) - segment_start_dist))
            end_point_index = np.argmin(np.abs(np.array(dists) - segment_end_dist))
            
            sub_segment_points = points[start_point_index:end_point_index + 1]
            
            if not sub_segment_points or len(sub_segment_points) < 2:
                current_sub_dist_from_pa_start = segment_end_dist - start_dist_pa
                continue
                
            # Calculate metrics for the sub-segment
            gain = 0
            loss = 0
            for j in range(1, len(sub_segment_points)):
                elevation_change = sub_segment_points[j].elevation - sub_segment_points[j-1].elevation
                if elevation_change > 0:
                    gain += elevation_change
                else:
                    loss += abs(elevation_change)
            
            segment_horizontal_dist = sub_segment_points[-1].distance_2d(sub_segment_points[0])
            slope = (gain - loss) / segment_horizontal_dist * 100 if segment_horizontal_dist > 0 else 0
            
            cumulative_gain += gain
            cumulative_loss += loss
            
            results.append({
                'dist_origen_km': (segment_end_dist) / 1000,
                'altitud_ganada': gain,
                'altitud_perdida': loss,
                'pendiente': slope,
                'altitud_ganada_acumulada': cumulative_gain,
                'altitud_perdida_acumulada': cumulative_loss,
                'distance_meters': segment_end_dist - segment_start_dist
            })
            
            current_sub_dist_from_pa_start = segment_end_dist - start_dist_pa
    return results

def format_seconds(seconds):
    """Helper function to format seconds into hh:mm:ss format."""
    total_hours = int(seconds // 3600)
    total_remaining_seconds = seconds % 3600
    total_minutes = int(total_remaining_seconds // 60)
    total_seconds = int(total_remaining_seconds % 60)
    return f"{total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}"

def calculate_time_metrics(pace_data, pa_locations, segment_data, rest_stops):
    """
    Calculates cumulative time metrics based on segment pace, distance, and rest stops.
    This version correctly handles time calculation up to the exact point of a rest stop.
    """
    time_results = []
    cumulative_time_seconds = 0
    time_since_last_pa_seconds = 0
    last_event_dist_km = 0
    
    # Combine all unique event distances (pace changes, PAs, and rest stops)
    event_distances = sorted(list(set(
        [0.0] +
        [item['distance'] for item in pace_data.values()] +
        [pa for pa in pa_locations] +
        [rest['dist_km'] for rest in rest_stops]
    )))
    
    for i in range(1, len(event_distances)):
        current_dist_km = event_distances[i]
        prev_dist_km = event_distances[i-1]
        
        dist_traveled = current_dist_km - prev_dist_km
        
        if dist_traveled <= 0:
            continue
            
        # Determine the pace for this segment
        pace_of_current_segment = '--:--'
        for seg in sorted(pace_data.values(), key=lambda x: x['distance']):
            if seg['distance'] >= current_dist_km:
                pace_of_current_segment = seg['pace']
                break
                
        try:
            minutes, seconds = map(int, pace_of_current_segment.split(':'))
            pace_seconds_per_km = minutes * 60 + seconds
            travel_time_seconds = pace_seconds_per_km * dist_traveled
        except (ValueError, IndexError):
            travel_time_seconds = 0
            
        cumulative_time_seconds += travel_time_seconds
        time_since_last_pa_seconds += travel_time_seconds
        
        pa_label = None
        for pa_index, pa_dist in enumerate(pa_locations):
            if abs(pa_dist - current_dist_km) < 0.01:
                pa_label = f"PA {pa_index + 1}"
                break
        
        rest_stop_time_seconds = 0
        rest_stop_match = next((rest for rest in rest_stops if abs(rest['dist_km'] - current_dist_km) < 0.01), None)
        if rest_stop_match:
            rest_stop_time_seconds = rest_stop_match['time_min'] * 60

        # Create a result entry for the travel segment
        time_results.append({
            'dist_origen_km': round(current_dist_km, 2),
            'type': pa_label if pa_label else 'Segmento',
            'total_time_formatted': format_seconds(cumulative_time_seconds),
            'cumulative_time_seconds': cumulative_time_seconds,
            'pa_time_formatted': format_seconds(time_since_last_pa_seconds),
            'time_since_last_pa_seconds': time_since_last_pa_seconds,
            'pace_formatted': pace_of_current_segment
        })
        
        # Add a separate entry for the rest time if it exists
        if rest_stop_time_seconds > 0:
            cumulative_time_seconds += rest_stop_time_seconds
            time_since_last_pa_seconds += rest_stop_time_seconds
            
            time_results.append({
                'dist_origen_km': round(current_dist_km, 2),
                'type': f"Parada en {pa_label}" if pa_label else 'Parada',
                'total_time_formatted': format_seconds(cumulative_time_seconds),
                'cumulative_time_seconds': cumulative_time_seconds,
                'pa_time_formatted': format_seconds(time_since_last_pa_seconds),
                'time_since_last_pa_seconds': time_since_last_pa_seconds,
                'pace_formatted': '--'
            })
            
        if pa_label:
            time_since_last_pa_seconds = 0
            
    return time_results

def calculate_pa_summary(pa_locations, segment_data, time_results):
    """
    Calculates a summary of metrics for each aid station (PA) and the end of the route.
    """
    summary_data = []
    
    # Use 0 as the starting point for "since last PA" calculations
    last_pa_dist_km = 0
    last_pa_time_s = 0
    last_pa_gain = 0
    last_pa_loss = 0

    # Combine PA locations and the final point of the route
    all_points_km = sorted(list(set(pa_locations + [segment_data[-1]['dist_origen_km']])))
    
    for i, point_dist_km in enumerate(all_points_km):
        
        # New robust way to check if the point is a PA
        pa_label = "Final"
        for pa_index, pa_dist in enumerate(pa_locations):
            if abs(pa_dist - point_dist_km) < 0.01:
                pa_label = f"PA {pa_index + 1}"
                break
        
        # Find the segment and time data at this point
        segment_entry = next((seg for seg in segment_data if abs(seg['dist_origen_km'] - point_dist_km) < 0.01), None)
        # Find the last time entry at or before the current point
        time_entry = None
        for res in time_results:
            if abs(res['dist_origen_km'] - point_dist_km) < 0.01 or res['dist_origen_km'] < point_dist_km:
                time_entry = res
        
        if not segment_entry or not time_entry:
            continue
            
        # Calculate metrics since the last PA or start
        dist_since_last_pa = point_dist_km - last_pa_dist_km
        time_since_last_pa_s = time_entry['cumulative_time_seconds'] - last_pa_time_s
        
        gain_since_last_pa = segment_entry['altitud_ganada_acumulada'] - last_pa_gain
        loss_since_last_pa = segment_entry['altitud_perdida_acumulada'] - last_pa_loss
        
        summary_data.append({
            'label': pa_label,
            'dist_total_km': round(point_dist_km, 2),
            'tiempo_total_s': format_seconds(time_entry['cumulative_time_seconds']),
            'dist_desde_pa_km': round(dist_since_last_pa, 2),
            'tiempo_desde_pa_s': format_seconds(time_since_last_pa_s),
            'alt_pos_total': round(segment_entry['altitud_ganada_acumulada'], 2),
            'alt_pos_desde_pa': round(gain_since_last_pa, 2),
            'alt_neg_total': round(segment_entry['altitud_perdida_acumulada'], 2),
            'alt_neg_desde_pa': round(loss_since_last_pa, 2),
        })

        # Update last PA values for the next iteration
        last_pa_dist_km = point_dist_km
        last_pa_time_s = time_entry['cumulative_time_seconds']
        last_pa_gain = segment_entry['altitud_ganada_acumulada']
        last_pa_loss = segment_entry['altitud_perdida_acumulada']
    
    return summary_data

@app.route('/', methods=['GET', 'POST'])
def handle_form():
    """Handles the GPX file upload and the generation of plots and the table."""
    
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'calculate_times':
            pace_data = {}
            for key, value in request.form.items():
                if key.startswith('pace_'):
                    index = int(key.split('_')[1])
                    dist_key = f'dist_{index}'
                    dist_from_origin_km = float(request.form.get(dist_key, 0))
                    pace_data[key] = {'pace': value, 'distance': dist_from_origin_km}
            
            # Get rest stop data
            rest_stops = []
            for key, value in request.form.items():
                if key.startswith('rest_stop_km_'):
                    index = key.split('_')[-1]
                    try:
                        dist_km = float(value)
                        time_min = float(request.form.get(f'rest_stop_min_{index}', 0))
                        if dist_km > 0 and time_min > 0:
                            rest_stops.append({'dist_km': dist_km, 'time_min': time_min})
                    except (ValueError, TypeError):
                        pass

            pa_locations_str = request.form.get('pa_locations', '[]')
            sub_segment_length_str = request.form.get('sub_segment_length', '0')
            segment_data_str = request.form.get('segment_data', '[]')

            # Parse JSON strings to Python objects
            pa_locations_km = json.loads(pa_locations_str)
            sub_segment_length = int(sub_segment_length_str)
            segment_data = json.loads(segment_data_str)
            
            altimetry_url = request.form.get('altimetry_plot_url')
            map_url = request.form.get('map_plot_url')

            time_results = calculate_time_metrics(pace_data, pa_locations_km, segment_data, rest_stops)
            pa_summary = calculate_pa_summary(pa_locations_km, segment_data, time_results)

            return render_template('results.html',
                                   altimetry_plot_url=altimetry_url,
                                   map_plot_url=map_url,
                                   segment_data=segment_data,
                                   pa_locations=pa_locations_km,
                                   sub_segment_length=sub_segment_length,
                                   time_results=time_results,
                                   pa_summary=pa_summary)

        else:
            file = request.files.get('gpx_file')
            
            if not file or not file.filename.endswith('.gpx'):
                return "Error: Por favor, sube un archivo GPX válido (.gpx).", 400

            try:
                gpx_data = gpxpy.parse(file.stream)
                
                total_distance_m = 0
                if gpx_data.tracks and gpx_data.tracks[0].segments:
                    segment = gpx_data.tracks[0].segments[0]
                    total_distance_m = segment.length_2d()
                
                total_distance_km = total_distance_m / 1000
                
                if total_distance_km == 0:
                    return "Error: No se pudo determinar la distancia del recorrido. El archivo GPX podría no contener datos de movimiento.", 400

                pa_count = int(request.form.get('pa_count', 0))
                pa_locations_km = []
                for i in range(1, pa_count + 1):
                    pa_location = float(request.form.get(f'pa_location_{i}', 0))
                    if 0 <= pa_location <= total_distance_km:
                        pa_locations_km.append(pa_location)
                    else:
                        return f"Error: La ubicación del PA {i} ({pa_location} km) está fuera del recorrido (0 km a {total_distance_km:.2f} km).", 400

                sub_segment_length = int(request.form.get('longitud_segmento', 0))
                if sub_segment_length <= 0:
                     return "Error: La longitud del sub-segmento debe ser un número entero positivo mayor que cero.", 400

                # Generate and save plots
                altimetry_buffer = generate_plot(gpx_data, pa_locations_km, 'altimetry')
                map_buffer = generate_plot(gpx_data, pa_locations_km, 'map')
                
                alt_img_path = os.path.join(app.config['STATIC_FOLDER'], 'altimetry.png')
                map_img_path = os.path.join(app.config['STATIC_FOLDER'], 'map.png')
                
                with open(alt_img_path, 'wb') as f:
                    f.write(altimetry_buffer.read())
                
                with open(map_img_path, 'wb') as f:
                    f.write(map_buffer.read())
                
                segment_data = analyze_segments(gpx_data, pa_locations_km, sub_segment_length)
                
                return render_template('results.html',
                                       altimetry_plot_url=url_for('static', filename='plots/altimetry.png'),
                                       map_plot_url=url_for('static', filename='plots/map.png'),
                                       segment_data=segment_data,
                                       pa_locations=pa_locations_km,
                                       sub_segment_length=sub_segment_length)

            except Exception as e:
                return f"Ocurrió un error al procesar el archivo: {e}", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

