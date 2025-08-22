import os
import io
import gpxpy
import gpxpy.gpx
import json
from flask import Flask, render_template, request, url_for, session
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import folium

# Se configura el backend de Matplotlib para poder generar gráficos
matplotlib.use('Agg')

# App configuration
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_session'
# Se define el directorio donde se guardarán los gráficos
app.config['STATIC_FOLDER'] = os.path.join(app.root_path, 'static', 'plots')
# Asegurarse de que el directorio de gráficos exista
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def generate_plot(gpx_data, pa_locations, plot_type='altimetry', race_name='Recorrido'):
    """
    Genera un gráfico de altimetría o un mapa 2D y lo devuelve como un buffer de bytes.
    El parámetro `race_name` se usa para el título del gráfico.
    """
    if not gpx_data.tracks or not gpx_data.tracks[0].segments:
        raise ValueError("El archivo GPX no contiene datos de tracks o segmentos.")
        
    points = gpx_data.tracks[0].segments[0].points
    
    # Extraer los datos de los puntos
    lats = [p.latitude for p in points]
    lons = [p.longitude for p in points]
    alts = [p.elevation for p in points]
    
    # Calcular la distancia acumulada de los puntos
    dists = [0]
    for i in range(1, len(points)):
        d = points[i].distance_2d(points[i-1])
        dists.append(dists[-1] + d)
    
    dists_km = [d / 1000 for d in dists]
    
    # Configurar el estilo del gráfico
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
        # Usamos el nombre de la carrera en el título del gráfico
        ax.set_title(f'Perfil de Altimetría - {race_name}', fontsize=16, fontweight='bold')
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
        # Usamos el nombre de la carrera en el título del mapa
        ax.set_title(f'Plano 2D del Circuito - {race_name}', fontsize=16, fontweight='bold')
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
    
def create_osm_map(gpx_data, pa_locations):
    """
    Genera un mapa interactivo con la ruta y los marcadores de PAs.
    
    Args:
        gpx_data (gpxpy.gpx.GPX): Objeto GPX analizado.
        pa_locations (list): Lista de ubicaciones de PA en KM.
        
    Returns:
        str: Ruta al archivo HTML del mapa.
    """
    if not gpx_data.tracks or not gpx_data.tracks[0].segments:
        return None
        
    points = gpx_data.tracks[0].segments[0].points
    coords = [(p.latitude, p.longitude) for p in points]

    # Crear mapa centrado en el primer punto
    osm_map = folium.Map(location=coords[0], zoom_start=13)

    # Dibujar la ruta
    folium.PolyLine(coords, color="blue", weight=3).add_to(osm_map)

    # Marcar inicio y fin
    folium.Marker(coords[0], tooltip="Inicio", icon=folium.Icon(color="green")).add_to(osm_map)
    folium.Marker(coords[-1], tooltip="Fin", icon=folium.Icon(color="red")).add_to(osm_map)

    # --- INICIO DE LA NUEVA LÓGICA PARA MARCADORES DE PA ---
    # Calcular la distancia acumulada de los puntos
    dists = [0]
    for i in range(1, len(points)):
        d = points[i].distance_2d(points[i-1])
        dists.append(dists[-1] + d)
    dists_km = [d / 1000 for d in dists]

    # Añadir marcadores para cada PA
    for i, pa_dist_km in enumerate(pa_locations):
        # Encontrar el punto GPX más cercano a la distancia del PA
        closest_point_index = np.argmin(np.abs(np.array(dists_km) - pa_dist_km))
        closest_point = points[closest_point_index]
        
        pa_coords = (closest_point.latitude, closest_point.longitude)
        
        # Añadir un marcador de círculo con un popup que muestre la distancia
        folium.Marker(
            location=pa_coords,
            popup=f"PA {i+1} ({pa_dist_km} km)",
            icon=folium.Icon(color="orange", icon="info-sign")
        ).add_to(osm_map)
    # --- FIN DE LA NUEVA LÓGICA ---

    # Guardar en la carpeta static
    map_path = os.path.join(app.config['STATIC_FOLDER'], 'osm_map.html')
    osm_map.save(map_path)
    return map_path

def analyze_segments(gpx_data, pa_locations, sub_segment_length):
    """
    Analiza y calcula métricas para sub-segmentos entre los PAs.
    
    Args:
        gpx_data (gpxpy.gpx.GPX): Objeto GPX analizado.
        pa_locations (list): Lista de ubicaciones de PA en KM.
        sub_segment_length (int): Longitud de cada sub-segmento en metros.
        
    Returns:
        list: Una lista de diccionarios con métricas para cada sub-segmento.
    """
    import math
    
    if not gpx_data.tracks or not gpx_data.tracks[0].segments:
        return []

    points = gpx_data.tracks[0].segments[0].points
    
    # Distancia acumulada
    dists = [0]
    for i in range(1, len(points)):
        d = points[i].distance_2d(points[i-1])
        dists.append(dists[-1] + d)
    
    # PAs en metros
    pa_locations_m = [loc * 1000 for loc in pa_locations]
    pa_locations_m.insert(0, 0)
    pa_locations_m.append(dists[-1])
    
    results = []
    cumulative_gain = 0
    cumulative_loss = 0
    
    # Iterar entre PAs
    for i in range(len(pa_locations_m) - 1):
        start_dist_pa = pa_locations_m[i]
        end_dist_pa = pa_locations_m[i+1]
        
        # Índices de inicio y fin
        start_index_pa = np.argmin(np.abs(np.array(dists) - start_dist_pa))
        end_index_pa = np.argmin(np.abs(np.array(dists) - end_dist_pa))

        # Sub-segmentos dentro del PA
        current_sub_dist_from_pa_start = 0
        while current_sub_dist_from_pa_start < (end_dist_pa - start_dist_pa):
            segment_start_dist = start_dist_pa + current_sub_dist_from_pa_start
            segment_end_dist = min(start_dist_pa + current_sub_dist_from_pa_start + sub_segment_length, end_dist_pa)

            # Puntos del sub-segmento
            start_point_index = np.argmin(np.abs(np.array(dists) - segment_start_dist))
            end_point_index = np.argmin(np.abs(np.array(dists) - segment_end_dist))
            sub_segment_points = points[start_point_index:end_point_index + 1]
            
            if not sub_segment_points or len(sub_segment_points) < 2:
                current_sub_dist_from_pa_start = segment_end_dist - start_dist_pa
                continue

            # Calcular ganancia y pérdida
            gain = 0
            loss = 0
            for j in range(1, len(sub_segment_points)):
                elevation_change = sub_segment_points[j].elevation - sub_segment_points[j-1].elevation
                if elevation_change > 0:
                    gain += elevation_change
                else:
                    loss += abs(elevation_change)
            
            # Altura máxima y mínima
            alts_segment = [p.elevation for p in sub_segment_points]
            max_alt = max(alts_segment)
            min_alt = min(alts_segment)
            delta_h = max_alt - min_alt

            # Distancia horizontal
            segment_horizontal_dist = sub_segment_points[-1].distance_2d(sub_segment_points[0])

            # Pendientes
            pendiente_promedio_pct = (gain - loss) / segment_horizontal_dist * 100 if segment_horizontal_dist > 0 else 0
            pendiente_max_angulo = math.degrees(math.atan(delta_h / segment_horizontal_dist)) if segment_horizontal_dist > 0 else 0

            cumulative_gain += gain
            cumulative_loss += loss

            results.append({
                'dist_origen_km': (segment_end_dist) / 1000,
                'altitud_ganada': gain,
                'altitud_perdida': loss,
                'pendiente_promedio_pct': pendiente_promedio_pct,
                'pendiente_max_angulo': pendiente_max_angulo,
                'altitud_ganada_acumulada': cumulative_gain,
                'altitud_perdida_acumulada': cumulative_loss,
                'distance_meters': segment_end_dist - segment_start_dist
            })
            
            current_sub_dist_from_pa_start = segment_end_dist - start_dist_pa
    
    return results

def format_seconds(seconds):
    """Función de ayuda para formatear segundos a formato hh:mm:ss."""
    total_hours = int(seconds // 3600)
    total_remaining_seconds = seconds % 3600
    total_minutes = int(total_remaining_seconds // 60)
    total_seconds = int(total_remaining_seconds % 60)
    return f"{total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}"

def calculate_time_metrics(pace_data, pa_locations, segment_data, rest_stops):
    """
    Calcula métricas de tiempo acumulado basadas en el ritmo del segmento, la distancia y las paradas.
    Esta versión maneja correctamente el cálculo del tiempo hasta el punto exacto de una parada.
    """
    time_results = []
    cumulative_time_seconds = 0
    time_since_last_pa_seconds = 0
    last_event_dist_km = 0
    
    # Combinar todas las distancias de eventos únicos (cambios de ritmo, PAs y paradas)
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
            
        # Determinar el ritmo para este segmento
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

        # Crear una entrada de resultado para el segmento de viaje
        time_results.append({
            'dist_origen_km': round(current_dist_km, 2),
            'type': pa_label if pa_label else 'Segmento',
            'total_time_formatted': format_seconds(cumulative_time_seconds),
            'cumulative_time_seconds': cumulative_time_seconds,
            'pa_time_formatted': format_seconds(time_since_last_pa_seconds),
            'time_since_last_pa_seconds': time_since_last_pa_seconds,
            'pace_formatted': pace_of_current_segment
        })
        
        # Añadir una entrada separada para el tiempo de descanso si existe
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
    Calcula un resumen de métricas para cada puesto de abastecimiento (PA) y el final de la ruta.
    """
    summary_data = []
    
    # Usar 0 como punto de partida para los cálculos "desde el último PA"
    last_pa_dist_km = 0
    last_pa_time_s = 0
    last_pa_gain = 0
    last_pa_loss = 0

    # Combinar las ubicaciones de PA y el punto final de la ruta
    all_points_km = sorted(list(set(pa_locations + [segment_data[-1]['dist_origen_km']])))
    
    for i, point_dist_km in enumerate(all_points_km):
        
        # Nueva forma robusta de verificar si el punto es un PA
        pa_label = "Final"
        for pa_index, pa_dist in enumerate(pa_locations):
            if abs(pa_dist - point_dist_km) < 0.01:
                pa_label = f"PA {pa_index + 1}"
                break
        
        # Encontrar los datos de segmento y tiempo en este punto
        segment_entry = next((seg for seg in segment_data if abs(seg['dist_origen_km'] - point_dist_km) < 0.01), None)
        # Encontrar la última entrada de tiempo en o antes del punto actual
        time_entry = None
        for res in time_results:
            if abs(res['dist_origen_km'] - point_dist_km) < 0.01 or res['dist_origen_km'] < point_dist_km:
                time_entry = res
        
        if not segment_entry or not time_entry:
            continue
            
        # Calcular métricas desde el último PA o el inicio
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

        # Actualizar los valores del último PA para la siguiente iteración
        last_pa_dist_km = point_dist_km
        last_pa_time_s = time_entry['cumulative_time_seconds']
        last_pa_gain = segment_entry['altitud_ganada_acumulada']
        last_pa_loss = segment_entry['altitud_perdida_acumulada']
    
    return summary_data

@app.route('/', methods=['GET', 'POST'])
def handle_form():
    """Maneja la subida del archivo GPX y la generación de gráficos y la tabla."""
    
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
            
            # Obtener datos de las paradas
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
            race_name_str = request.form.get('race_name', 'Resultados')

            # Analizar las cadenas JSON a objetos de Python
            pa_locations_km = json.loads(pa_locations_str)
            sub_segment_length = int(sub_segment_length_str)
            segment_data = json.loads(segment_data_str)
            
            altimetry_url = request.form.get('altimetry_plot_url')
            map_url = request.form.get('map_plot_url')

            time_results = calculate_time_metrics(pace_data, pa_locations_km, segment_data, rest_stops)
            pa_summary = calculate_pa_summary(pa_locations_km, segment_data, time_results)

            # La llamada a create_osm_map se ha movido al bloque de subida del archivo.
            # No la necesitamos aquí, pero aún necesitamos la URL del mapa OSM
            # para pasarla a la plantilla. La URL está en la sesión.
            osm_map_url = session.get('osm_map_url', '#')

            return render_template('results.html',
                                   race_name=race_name_str,
                                   altimetry_plot_url=altimetry_url,
                                   map_plot_url=map_url,
                                   osm_map_url=osm_map_url,
                                   segment_data=segment_data,
                                   pa_locations=pa_locations_km,
                                   sub_segment_length=sub_segment_length,
                                   time_results=time_results,
                                   pa_summary=pa_summary)

        else:
            # Capturar el nombre de la carrera del formulario
            race_name = request.form.get('race_name', 'Resultados de Carrera')
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

                # Paso 1: Generar el mapa de altimetría y el mapa 2D
                altimetry_buffer = generate_plot(gpx_data, pa_locations_km, 'altimetry', race_name)
                map_buffer = generate_plot(gpx_data, pa_locations_km, 'map', race_name)
                
                alt_img_path = os.path.join(app.config['STATIC_FOLDER'], 'altimetry.png')
                map_img_path = os.path.join(app.config['STATIC_FOLDER'], 'map.png')
                
                with open(alt_img_path, 'wb') as f:
                    f.write(altimetry_buffer.read())
                
                with open(map_img_path, 'wb') as f:
                    f.write(map_buffer.read())

                # Paso 2: Generar el mapa OSM y guardar su ruta en la sesión
                # La llamada ahora incluye la lista de ubicaciones de PA
                osm_map_path = create_osm_map(gpx_data, pa_locations_km)
                session['osm_map_url'] = url_for('static', filename='plots/osm_map.html')
                
                segment_data = analyze_segments(gpx_data, pa_locations_km, sub_segment_length)
                
                return render_template('results.html',
                                       race_name=race_name, # El nombre de la carrera se pasa a la plantilla
                                       altimetry_plot_url=url_for('static', filename='plots/altimetry.png'),
                                       map_plot_url=url_for('static', filename='plots/map.png'),
                                       osm_map_url=session.get('osm_map_url'), # Ahora se obtiene de la sesión
                                       segment_data=segment_data,
                                       pa_locations=pa_locations_km,
                                       sub_segment_length=sub_segment_length)

            except Exception as e:
                return f"Ocurrió un error al procesar el archivo: {e}", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

