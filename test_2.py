
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.linalg import eigh
from obspy import Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from random import uniform

"""
MODIFICACIONES PARA DETECCIÓN DE DOS FUENTES:
1. Generación de señal:
   - Añadidos parámetros true_azimuth2 y true_slowness_ms2 en métodos:
     * __generate_test_signal()
     * get_test_traces()
     * run_music_test()
   - Suma de dos señales con retardos independientes en los sensores

2. Visualización:
   - Gráfico con dos picos identificados
   - Panel de resultados mostrando para cada fuente:
     * Backazimuth y slowness estimados/verdaderos
     * Erros de estimación
   - Salida por consola de los parámetros estimados

4. Otros:
   - Manejo de casos con 0, 1 o 2 fuentes detectadas
   - Cálculo independiente para cada fuente
"""

class ArrayToolsTest:
    def __init__(self):
        self.st = None
        self.coords = {}
        self.data_array = None
        self.index_list = []

    def __morlet_wavelet(self, npts, fs, w=6.0, plot=False):
        t = np.arange(-1*npts/fs, npts/fs, 1/fs)
        wavelet = np.exp(1j * w * t) * np.exp(-t ** 2 / 2)
        wavelet = wavelet * (np.pi ** (-0.25))
        wavelet = np.roll(wavelet, int(npts)//2)
        half_point = int(npts)
        wavelet = wavelet[half_point:-1]
        return wavelet.real

    def __plot_array_geometry(self, max_aperture_km=None, show_labels=True):
        plt.figure(figsize=(8, 8))
        for sensor_id, (x, y) in self.coords.items():
            plt.plot(x, y, 'o', markersize=8, label=sensor_id if show_labels else "")
            if show_labels:
                plt.text(x + 0.1, y + 0.1, sensor_id.split('.')[1], fontsize=9)

        if max_aperture_km:
            circle = plt.Circle((0, 0), max_aperture_km, color='gray', fill=False, linestyle='--', linewidth=1)
            plt.gca().add_artist(circle)

        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.xlabel("Easting [km]")
        plt.ylabel("Northing [km]")
        plt.title("Sensor Array Geometry")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    def __is_valid_position(self, new_pos, existing_positions, min_distance_km):
        for pos in existing_positions:
            if np.linalg.norm(np.array(new_pos) - np.array(pos)) < min_distance_km:
                return False
        return True

    def __create_test_array(self, num_sensors=10, max_aperture_km=5.0, min_distance_km=0.5):
        center_lat, center_lon = 40.0, -3.0
        synthetic_traces = []
        positions = []
        attempts = 0
        max_attempts = 1000

        while len(positions) < num_sensors and attempts < max_attempts:
            r = uniform(0, max_aperture_km)
            theta = uniform(0, 2 * np.pi)
            x_km = r * np.cos(theta)
            y_km = r * np.sin(theta)
            candidate = (x_km, y_km)

            if self.__is_valid_position(candidate, positions, min_distance_km):
                positions.append(candidate)
            attempts += 1

        if len(positions) < num_sensors:
            raise RuntimeError("Couldn't place all sensors with the given minimum distance.")

        self.coords = {}
        for i, (x_km, y_km) in enumerate(positions):
            lat_offset = x_km / 110.574
            lon_offset = y_km / (111.320 * np.cos(np.radians(center_lat)))
            lat = center_lat + lat_offset
            lon = center_lon + lon_offset

            sensor_id = f"SY.ST{i:02d}..SHZ"
            self.coords[sensor_id] = [x_km, y_km]

            stats = {
                'network': 'SY', 'station': f'ST{i:02d}', 'location': '', 'channel': 'SHZ',
                'npts': 5000, 'sampling_rate': 5000, 'starttime': UTCDateTime.now(),
                'coordinates': AttribDict({'latitude': lat, 'longitude': lon, 'elevation': 0.0})
            }
            tr = Trace(data=np.zeros(stats['npts']), header=stats)
            synthetic_traces.append(tr)

        self.st = Stream(synthetic_traces)
        self.index_list = list(self.coords.keys())
        self.__plot_array_geometry()
        print(f"Geometría del array (posición aleatoria):")
        for tr in self.st:
            x, y = self.coords[tr.id]
            print(f"{tr.id}: Este={x:.3f} km, Norte={y:.3f} km")

    def __make_steering(self, positions, az, slowness, freq):
        az_rad = np.deg2rad(az)
        direction = np.array([np.cos(az_rad), np.sin(az_rad)])
        delays = positions @ direction * slowness
        return np.exp(-2j * np.pi * freq * delays)

    def __create_fixed_array(self):
        center_lat, center_lon = 40.0, -3.0
        spacing_km = 2.0
        grid_shape = (5, 4)
        max_aperture_km = 10.0
        radius_limit = max_aperture_km / 2

        synthetic_traces = []
        self.coords = {}
        sensor_index = 0

        x_offsets = np.linspace(-spacing_km * (grid_shape[0] - 1) / 2,
                                spacing_km * (grid_shape[0] - 1) / 2, grid_shape[0])
        y_offsets = np.linspace(-spacing_km * (grid_shape[1] - 1) / 2,
                                spacing_km * (grid_shape[1] - 1) / 2, grid_shape[1])

        for x in x_offsets:
            for y in y_offsets:
                if np.sqrt(x ** 2 + y ** 2) <= radius_limit:
                    sensor_id = f"SY.ST{sensor_index:02d}..SHZ"
                    self.coords[sensor_id] = [x, y]

                    lat_offset = y / 110.574
                    lon_offset = x / (111.320 * np.cos(np.radians(center_lat)))
                    lat = center_lat + lat_offset
                    lon = center_lon + lon_offset

                    stats = {
                        'network': 'SY', 'station': f'ST{sensor_index:02d}', 'location': '', 'channel': 'SHZ',
                        'npts': 5000, 'sampling_rate': 5000, 'starttime': UTCDateTime.now(),
                        'coordinates': AttribDict({'latitude': lat, 'longitude': lon, 'elevation': 0.0})
                    }
                    tr = Trace(data=np.zeros(stats['npts']), header=stats)
                    synthetic_traces.append(tr)

                    sensor_index += 1
                    if sensor_index == 20:
                        break
            if sensor_index == 20:
                break

        self.st = Stream(synthetic_traces)
        self.index_list = list(self.coords.keys())
        print(f"Fixed Array Geometry (max aperture: 10 km, min spacing: 5 km):")
        for tr in self.st:
            x, y = self.coords[tr.id]
            print(f"{tr.id}: East={x:.2f} km, North={y:.2f} km")

    def __convert_to_array(self):
        n = len(self.st)
        if n == 0:
            raise ValueError("No hay trazas para convertir")
        npts = self.st[0].stats.npts
        self.data_array = np.zeros((n, npts), dtype=complex)
        for i, tr in enumerate(self.st):
            if len(tr.data) != npts:
                raise ValueError(f"Todas las trazas deben tener igual longitud. Traza {tr.id} tiene {len(tr.data)} muestras")
            self.data_array[i, :] = hilbert(tr.data)
        print(f"Array convertido: {self.data_array.shape} (sensores x muestras)")

    def __generate_test_signal(self, fs, fc, true_azimuth1=45.0, true_slowness_ms1=250.0, 
                             true_azimuth2=None, true_slowness_ms2=None):
        T = 10 / fc
        npts = int(T * fs)
        base_wave = self.__morlet_wavelet(npts, fs, 6.0)

        true_slowness1 = true_slowness_ms1 / 1000
        print(f"Generando señal 1 con azimuth={true_azimuth1}°, slowness={true_slowness1:.6f} s/km")

        positions = np.array([self.coords[tr.id] for tr in self.st])
        print("Posiciones de los sensores (km):", positions)
        
        # Generar primera señal
        for tr in self.st:
            x_km, y_km = self.coords[tr.id]
            az_rad = np.deg2rad(true_azimuth1)
            direction = np.array([np.cos(az_rad), np.sin(az_rad)])
            delay = (x_km * direction[0] + y_km * direction[1]) * true_slowness1
            delay_samples = int(round(delay * fs))
            tr.data = np.roll(base_wave, delay_samples)
            tr.stats.sampling_rate = 100
            tr.stats.delay = delay
        
        # Generar segunda señal si existe
        if true_azimuth2 is not None and true_slowness_ms2 is not None:
            true_slowness2 = true_slowness_ms2 / 1000
            print(f"Generando señal 2 con azimuth={true_azimuth2}°, slowness={true_slowness2:.6f} s/km")
            
            for tr in self.st:
                x_km, y_km = self.coords[tr.id]
                az_rad = np.deg2rad(true_azimuth2)
                direction = np.array([np.cos(az_rad), np.sin(az_rad)])
                delay = (x_km * direction[0] + y_km * direction[1]) * true_slowness2
                delay_samples = int(round(delay * fs))
                # Sumar la segunda señal a la existente
                second_signal = np.roll(base_wave, delay_samples)
                tr.data = tr.data + second_signal
                tr.stats.delay2 = delay

        self.st.plot()

        print("Demoras calculadas (s):", [f"{tr.stats.delay:.6f}" for tr in self.st])
        print("Muestras de delay:", [int(round(tr.stats.delay * fs)) for tr in self.st])
        print(f"Delay teórico para {tr.id}: {delay:.6f}s")

        true_dir = np.array([np.sin(np.deg2rad(true_azimuth1)), np.cos(np.deg2rad(true_azimuth1))])
        theoretical_delays = positions @ true_dir * true_slowness1
        print(f"[DEBUG] Verificación de retrasos:")
        print("Retrasos calculados vs generados:")
        for i, tr in enumerate(self.st):
            print(f"Sensor {tr.id}: Teórico={theoretical_delays[i]:.6f}s, Generado={tr.stats.delay:.6f}s")

    def get_test_traces(self, fs, fc, true_azimuth1=45.0, true_slowness_ms1=250.0,
                      true_azimuth2=None, true_slowness_ms2=None):
        print(f"get_test_traces: true_slowness_ms1={true_slowness_ms1}, true_slowness_ms2={true_slowness_ms2}")
        self.__generate_test_signal(fs, fc, true_azimuth1, true_slowness_ms1, 
                                  true_azimuth2, true_slowness_ms2)
        self.__convert_to_array()

    def __get_sensor_positions(self):
        sensor_positions = []
        for tr_id in self.index_list:
            x, y = self.coords[tr_id]
            sensor_positions.append([x, y])
            print(f"Sensor {tr_id}: Este={x:.3f} km, Norte={y:.3f} km")
        return np.array(sensor_positions)

    def __get_covariance_matrix(self, data):
        if data.ndim != 2:
            raise ValueError(f"Se esperaba array 2D, se recibió {data.ndim}D")
        R = np.dot(data, data.T.conj()) / data.shape[1]
        print("Matriz de covarianza (con Hilbert):", R)
        if np.any(np.isnan(R)) or np.any(np.isinf(R)):
            raise ValueError("Matriz de covarianza contiene NaN o infinitos")
        return R

    def __get_noise_subspace(self, R, n_signals=1):
        eigvals, eigvecs = eigh(R)
        print("Valores propios:", eigvals)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        signal_vec = eigvecs[:, :n_signals]
        print(f"Vector(es) propio(s) dominante(s) (subespacio de señal) para n_signals={n_signals}:", signal_vec)
        En = eigvecs[:, n_signals:]
        print("Shape of En:", En.shape)
        return En, signal_vec

    def __plot_slowness_map(self, music_map, azimuths, slowness_range,
                            peak_slowness1, peak_baz1,
                            peak_slowness2, peak_baz2,
                            true_azimuth1=None, true_slowness1=None,
                            true_azimuth2=None, true_slowness2=None):

        # Normalizar el mapa MUSIC
        music_map_db = 10 * np.log10(music_map / np.max(music_map) + 1e-12)
        music_map_db = np.clip(music_map_db, -5, 0)

        # Configurar figura
        plt.figure(figsize=(14, 6))

        # Gráfico completo
        ax1 = plt.subplot(121, polar=True)
        theta = np.deg2rad(azimuths)
        r, th = np.meshgrid(slowness_range, theta)
        c = ax1.contourf(th, r, music_map_db.T, levels=20, cmap='rainbow')
        plt.colorbar(c, ax=ax1, label='Power (dB)')
        ax1.set_title("Espectro MUSIC Completo", pad=20)

        # Marcar picos
        if peak_baz1 is not None:
            ax1.plot(np.deg2rad(peak_baz1), peak_slowness1, 'w*', markersize=15,
                     label=f'Pico 1: {peak_baz1:.1f}°, {peak_slowness1:.3f} s/km')
        if peak_baz2 is not None:
            ax1.plot(np.deg2rad(peak_baz2), peak_slowness2, 'ko', markersize=10,
                     label=f'Pico 2: {peak_baz2:.1f}°, {peak_slowness2:.3f} s/km')

        # Gráfico de información de parámetros
        ax2 = plt.subplot(122)
        ax2.set_title("Estimación de Parámetros", pad=20)
        ax2.axis('off')

        info_text = ""
        if peak_baz1 is not None:
            info_text += f"Fuente 1 estimada:\n"
            info_text += f"- Backazimuth: {peak_baz1:.1f}°\n"
            info_text += f"- Slowness: {peak_slowness1:.3f} s/km\n"
            if true_azimuth1 is not None:
                true_baz1 = (true_azimuth1 + 180) % 360
                info_text += f"- BAZ verdadero: {true_baz1:.1f}°\n"
                error_az = min(abs(peak_baz1 - true_baz1), 360 - abs(peak_baz1 - true_baz1))
                info_text += f"- Error azimuth: {error_az:.1f}°\n"
            if true_slowness1 is not None:
                info_text += f"- Slowness verdadero: {true_slowness1:.3f} s/km\n"
                error_slow = abs(peak_slowness1 - true_slowness1)
                info_text += f"- Error slowness: {error_slow:.3f} s/km\n"
            info_text += "\n"

        if peak_baz2 is not None:
            info_text += f"Fuente 2 estimada:\n"
            info_text += f"- Backazimuth: {peak_baz2:.1f}°\n"
            info_text += f"- Slowness: {peak_slowness2:.3f} s/km\n"
            if true_azimuth2 is not None:
                true_baz2 = (true_azimuth2 + 180) % 360
                info_text += f"- BAZ verdadero: {true_baz2:.1f}°\n"
                error_az = min(abs(peak_baz2 - true_baz2), 360 - abs(peak_baz2 - true_baz2))
                info_text += f"- Error azimuth: {error_az:.1f}°\n"
            if true_slowness2 is not None:
                info_text += f"- Slowness verdadero: {true_slowness2:.3f} s/km\n"
                error_slow = abs(peak_slowness2 - true_slowness2)
                info_text += f"- Error slowness: {error_slow:.3f} s/km\n"

        # Imprimir por consola
        print("\nRESULTADOS DE ESTIMACIÓN:")
        print(info_text)

        ax2.text(0.1, 0.5, info_text, fontsize=12, va='center')
        ax1.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def __compute_music_spectrum(self, positions, fs, En, signal_vec, slow_lim,
                                 true_azimuth1=None, true_slowness1=None,
                                 true_azimuth2=None, true_slowness2=None):
        azimuths = np.linspace(0, 360, 361)
        slowness_range = np.linspace(0.2, slow_lim, 500)
        freq = 1.0
        music_map = np.zeros((len(slowness_range), len(azimuths)))

        # Calcular mapa MUSIC completo
        for i, s in enumerate(slowness_range):
            for j, az in enumerate(azimuths):
                steering = self.__make_steering(positions, az, s, freq)[:, np.newaxis]
                steering = steering / np.linalg.norm(steering)
                proj = steering.T.conj() @ En @ En.T.conj() @ steering
                music_map[i, j] = 1 / (np.abs(proj.item()) + 1e-10)

        # Encontrar los picos
        from scipy.signal import find_peaks
        flattened = music_map.flatten()
        peaks, _ = find_peaks(flattened, height=np.mean(flattened), distance=10000)

        # Ordenar picos por altura
        peak_values = flattened[peaks]
        sorted_idx = np.argsort(peak_values)[::-1]
        peaks = peaks[sorted_idx]

        # Extraer hasta 2 picos más fuertes
        peak_slownesses = []
        peak_azimuths = []

        for i in range(min(2, len(peaks))):
            peak_idx = peaks[i]
            slow_idx = peak_idx // len(azimuths)
            az_idx = peak_idx % len(azimuths)
            peak_slownesses.append(slowness_range[slow_idx])
            peak_azimuths.append(azimuths[az_idx])

        # Asegurar que siempre hay 2 valores (rellenar con None si es necesario)
        while len(peak_slownesses) < 2:
            peak_slownesses.append(None)
            peak_azimuths.append(None)

        # Devolver 8 valores consistentemente
        return (music_map, music_map.copy(),  # Duplicado intencional para mantener compatibilidad
                azimuths, slowness_range,
                peak_slownesses[0], peak_azimuths[0],
                peak_slownesses[1], peak_azimuths[1])


    """
    def __compute_music_spectrum(self, positions, fs, En, signal_vec, slow_lim,
                                 true_azimuth1=None, true_slowness1=None,
                                 true_azimuth2=None, true_slowness2=None):
        azimuths = np.linspace(0, 360, 361)
        slowness_range = np.linspace(0.2, slow_lim, 500)
        freq = 1.0
        music_map = np.zeros((len(slowness_range), len(azimuths)))

        # Calcular mapa MUSIC completo
        for i, s in enumerate(slowness_range):
            for j, az in enumerate(azimuths):
                steering = self.__make_steering(positions, az, s, freq)[:, np.newaxis]
                steering = steering / np.linalg.norm(steering)
                proj = steering.T.conj() @ En @ En.T.conj() @ steering
                music_map[i, j] = 1 / (np.abs(proj.item()) + 1e-10)

        # Suavizado del espectro para mejor detección de picos
        from scipy.ndimage import gaussian_filter
        smoothed_map = gaussian_filter(music_map, sigma=3)

        # Encontrar picos con mayor separación angular
        def find_peaks_2d(map_data, min_distance=30):
            peaks = []
            for i in range(len(slowness_range)):
                for j in range(len(azimuths)):
                    if map_data[i, j] == np.max(map_data[max(0, i - 5):i + 5, max(0, j - 5):j + 5]):
                        if all(np.sqrt((i - x) ** 2 + ((j - y) * 2) ** 2) > min_distance for (x, y) in peaks):
                            peaks.append((i, j))
                            if len(peaks) == 2:
                                return peaks
            return peaks[:2]

        peaks = find_peaks_2d(smoothed_map, min_distance=30)

        # Ordenar picos por potencia
        if len(peaks) == 2:
            if smoothed_map[peaks[0]] < smoothed_map[peaks[1]]:
                peaks = [peaks[1], peaks[0]]

        # Extraer parámetros
        peak_slowness1 = slowness_range[peaks[0][0]] if len(peaks) > 0 else None
        peak_baz1 = azimuths[peaks[0][1]] if len(peaks) > 0 else None
        peak_slowness2 = slowness_range[peaks[1][0]] if len(peaks) > 1 else None
        peak_baz2 = azimuths[peaks[1][1]] if len(peaks) > 1 else None

        # Forzar separación si tenemos valores verdaderos
        if true_azimuth1 is not None and true_azimuth2 is not None:
            true_baz1 = (true_azimuth1 + 180) % 360
            true_baz2 = (true_azimuth2 + 180) % 360

            # Buscar picos cerca de las direcciones verdaderas
            def find_nearest_peak(true_baz):
                closest = (0, 0)
                min_dist = 360
                for i in range(len(slowness_range)):
                    for j in range(len(azimuths)):
                        dist = min(abs(azimuths[j] - true_baz), 360 - abs(azimuths[j] - true_baz))
                        if dist < min_dist and music_map[i, j] > 0.5 * np.max(music_map):
                            min_dist = dist
                            closest = (i, j)
                return closest

            if peak_baz1 is None or peak_baz2 is None:
                peak1 = find_nearest_peak(true_baz1)
                peak2 = find_nearest_peak(true_baz2)
                peak_slowness1 = slowness_range[peak1[0]]
                peak_baz1 = azimuths[peak1[1]]
                peak_slowness2 = slowness_range[peak2[0]]
                peak_baz2 = azimuths[peak2[1]]

        return (music_map, azimuths, slowness_range,
                peak_slowness1, peak_baz1,
                peak_slowness2, peak_baz2)
    """

    def run_music_test(self, true_azimuth1=45.0, true_slowness_ms1=250.0,
                       true_azimuth2=None, true_slowness_ms2=None,
                       fs=100, fc=1, slow_lim=0.3):
        true_slowness1 = true_slowness_ms1 / 1000
        print(f"true_slowness_ms1: {true_slowness_ms1}, true_slowness1: {true_slowness1:.6e} s/km")
        if true_azimuth2 is not None:
            true_slowness2 = true_slowness_ms2 / 1000
            print(f"true_slowness_ms2: {true_slowness_ms2}, true_slowness2: {true_slowness2:.6e} s/km")

        self.__create_test_array()
        self.get_test_traces(fs, fc, true_azimuth1, true_slowness_ms1,
                             true_azimuth2, true_slowness_ms2)
        data = self.data_array
        sensor_positions = self.__get_sensor_positions()
        fs = self.st[0].stats.sampling_rate
        R = self.__get_covariance_matrix(data)

        # Determinar número de señales a buscar
        n_signals = 2 if true_azimuth2 is not None else 1
        En, signal_vec = self.__get_noise_subspace(R, n_signals=n_signals)

        # Recibir los 8 valores devueltos
        (music_map1, music_map2, azimuths, slowness_range,
         peak_slowness1, peak_baz1, peak_slowness2, peak_baz2) = self.__compute_music_spectrum(
            sensor_positions, fs, En, signal_vec, slow_lim,
            true_azimuth1, true_slowness1, true_azimuth2, true_slowness2
        )

        #self.__plot_slowness_map(music_map1, azimuths, slowness_range,
        #                         peak_slowness1, peak_baz1, peak_slowness2, peak_baz2,
        #                         true_azimuth1, true_azimuth2)
        self.__plot_slowness_map(music_map1, azimuths, slowness_range,
                               peak_slowness1, peak_baz1, peak_slowness2, peak_baz2,
                               true_azimuth1, true_slowness1,
                               true_azimuth2, true_slowness2 if true_azimuth2 is not None else None)

if __name__ == "__main__":
    ATT = ArrayToolsTest()
    print("Creando datos de prueba...")
    # Ejemplo con una sola fuente
    # ATT.run_music_test(true_azimuth1=90.0, true_slowness_ms1=250.0, fs=100, fc=1)
    
    # Ejemplo con dos fuentes
    # Prueba 1
    #ATT.run_music_test(true_azimuth1=90.0, true_slowness_ms1=250.0,
    #                  true_azimuth2=180.0, true_slowness_ms2=350.0,
    #                  fs=100, fc=1)

    # Prueba 2
    #ATT.run_music_test(true_azimuth1=90.0, true_slowness_ms1=250.0,
    #                  true_azimuth2=270.0, true_slowness_ms2=350.0,
    #                  fs=100, fc=1)

    # Prueba 3  --> el resultado depende de la geometría, de los sensores, comprobar varias ejecuciones seguidas
    #               COMPROBAR GEOMETRÍA
    ATT.run_music_test(true_azimuth1=90.0, true_slowness_ms1=250.0,
                      true_azimuth2=330.0, true_slowness_ms2=275.0,
                      fs=100, fc=1)