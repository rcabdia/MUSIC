#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test



:param :
:type :
:return:
:rtype:
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.linalg import eigh
from obspy import Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from random import uniform

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
        t = t[half_point:-1]
        if plot:
            #Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(t, wavelet.real, label='Real part')


            plt.title('Morlet Wavelet')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return wavelet.real

    import matplotlib.pyplot as plt

    def __plot_array_geometry(self, max_aperture_km=None, show_labels=True):
        plt.figure(figsize=(8, 8))
        for sensor_id, (x, y) in self.coords.items():
            plt.plot(x, y, 'o', markersize=8, label=sensor_id if show_labels else "")
            if show_labels:
                plt.text(x + 0.1, y + 0.1, sensor_id.split('.')[1], fontsize=9)

        # Optional: draw maximum aperture circle
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

    def __create_fixed_array(self):
        center_lat, center_lon = 40.0, -3.0
        spacing_km = 2.0
        grid_shape = (5, 4)  # 5 columns (x), 4 rows (y)
        max_aperture_km = 10.0
        radius_limit = max_aperture_km / 2

        synthetic_traces = []
        self.coords = {}
        sensor_index = 0

        # Center grid around (0, 0)
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
        #self.__plot_array_geometry()
        print("\nFixed Array Geometry (max aperture: 10 km, min spacing: 5 km):")
        for tr in self.st:
            x, y = self.coords[tr.id]
            print(f"{tr.id}: East={x:.2f} km, North={y:.2f} km")
    def __create_test_array(self, num_sensors=10, max_aperture_km=5.0, min_distance_km=0.5):


        center_lat, center_lon = 40.0, -3.0
        synthetic_traces = []
        positions = []

        def is_valid_position(new_pos, existing_positions):
            for pos in existing_positions:
                if np.linalg.norm(np.array(new_pos) - np.array(pos)) < min_distance_km:
                    return False
            return True

        attempts = 0
        max_attempts = 1000  # to avoid infinite loops

        while len(positions) < num_sensors and attempts < max_attempts:
            # Random position in polar coordinates
            r = uniform(0, max_aperture_km)
            theta = uniform(0, 2 * np.pi)
            x_km = r * np.cos(theta)
            y_km = r * np.sin(theta)
            candidate = (x_km, y_km)

            if is_valid_position(candidate, positions):
                positions.append(candidate)
            attempts += 1

        if len(positions) < num_sensors:
            raise RuntimeError("Couldn't place all sensors with the given minimum distance.")

        self.coords = {}
        for i, (x_km, y_km) in enumerate(positions):
            # Convert km offsets to lat/lon
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
        print("\nGeometría del array (posición aleatoria):")
        for tr in self.st:
            x, y = self.coords[tr.id]
            print(f"{tr.id}: Este={x:.3f} km, Norte={y:.3f} km")

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

    def get_test_traces(self, fs, fc, true_azimuth=45.0, true_slowness_ms=250.0):

        print(f"get_test_traces: true_slowness_ms={true_slowness_ms}")
        self.__generate_test_signal(fs, fc, true_azimuth, true_slowness_ms)
        self.__convert_to_array()

    def __get_sensor_positions(self):
        sensor_positions = []
        for tr_id in self.index_list:
            x, y = self.coords[tr_id]
            sensor_positions.append([x, y])
            print(f"Sensor {tr_id}: Este={x:.3f} km, Norte={y:.3f} km")
        return np.array(sensor_positions)

    def __get_noise_subspace(self, R, n_signals=1):
        eigvals, eigvecs = eigh(R)
        print("Valores propios:", eigvals)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        signal_vec = eigvecs[:, :n_signals]
        print(f"Vector(es) propio(s) dominante(s) (subespacio de señal) para n_signals={n_signals}:\n", signal_vec)
        En = eigvecs[:, n_signals:]
        print("Shape of En:", En.shape)
        return En, signal_vec

    def __get_covariance_matrix(self, data):
        try:
            if data.ndim != 2:
                raise ValueError(f"Se esperaba array 2D, se recibió {data.ndim}D")
            R = np.dot(data, data.T.conj()) / data.shape[1]
            print("Matriz de covarianza (con Hilbert):\n", R)
            if np.any(np.isnan(R)) or np.any(np.isinf(R)):
                raise ValueError("Matriz de covarianza contiene NaN o infinitos")
            return R
        except Exception as e:
            print(f"Error en covarianza: {str(e)}")
            raise

    def __plot_slowness_map(self, music_map, azimuths, slowness_range, peak_slowness, peak_baz, true_azimuth=None):
        music_map = np.nan_to_num(music_map, nan=0.0, posinf=0.0, neginf=0.0)
        peak_theta_rad = np.deg2rad(azimuths)
        max_val = np.max(music_map)

        if max_val == 0:
            print("Advertencia: music_map contiene solo ceros")
            return

        music_map = 10 * np.log10(music_map / max_val + 1e-12)
        music_map = np.clip(music_map, a_min=-5.0, a_max=0)

        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, polar=True)
        r, theta = np.meshgrid(slowness_range, peak_theta_rad, indexing='ij')
        c = ax.contourf(theta, r, music_map, cmap='rainbow', levels=100, vmin=-5.0, vmax=0)

        # Mostrar siempre el BAZ correcto (azimuth + 180)
        #peak_baz = (peak_azimuth + 180) % 360
        print(f"Dibujando marcador en BAZ={peak_baz:.1f}°, slowness={peak_slowness:.4f}")

        ax.plot(np.deg2rad(peak_baz), peak_slowness, 'w*', markersize=15, label='Pico')
        plt.colorbar(c, label='Power dB')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title("MUSIC Spectrum - Test Array", color='black')

        if true_azimuth is not None:
            ax.legend([
                f'Pico estimado: {peak_baz:.1f}° (Verdadero: {(true_azimuth + 180) % 360:.1f}°)\nSlowness: {peak_slowness:.3f} s/km'
            ], loc='upper right')
        else:
            ax.legend([f'Peak BAZ: {peak_baz:.1f}°, S: {peak_slowness:.3f} s/km'], loc='upper right')

        plt.show()

    def __generate_test_signal(self, fs, fc, true_azimuth=45.0, true_slowness_ms=250.0):

        T = 10/fc
        npts = T*fs

        base_wave = self.__morlet_wavelet(npts, fs, 6.0).real


        true_slowness = true_slowness_ms / 1000
        print(f"\nGenerando señal con azimuth={true_azimuth}°, slowness={true_slowness:.6f} s/km")

        # Obtener posiciones para diagnóstico
        positions = np.array([self.coords[tr.id] for tr in self.st])
        print("Posiciones de los sensores (km):\n", positions)
        all_traces = []

        for tr in self.st:
            x_km, y_km = self.coords[tr.id]
            az_rad = np.deg2rad(true_azimuth)

            direction = np.array([np.cos(az_rad), np.sin(az_rad)])

            delay = (x_km * direction[0] + y_km * direction[1]) * true_slowness
            delay_samples = int(round(delay * fs))
            tr.data = np.roll(base_wave, delay_samples)
            tr.stats.sampling_rate = 100
            tr.stats.delay = delay
            all_traces.append(tr)


        self.st.plot()

        # Diagnóstico final
        print("Demoras calculadas (s):", [f"{tr.stats.delay:.6f}" for tr in self.st])
        print("Muestras de delay:", [int(round(tr.stats.delay * fs)) for tr in self.st])
        print(f"Delay teórico para {tr.id}: {delay:.6f}s")

        # ===== PRUEBA ADICIONAL 1: Verificación de retrasos =====
        print("\n[DEBUG] Verificación de retrasos:")
        true_dir = np.array([np.sin(np.deg2rad(true_azimuth)), np.cos(np.deg2rad(true_azimuth))])
        theoretical_delays = positions @ true_dir * true_slowness
        print("Retrasos calculados vs generados:")
        for i, tr in enumerate(self.st):
            print(f"Sensor {tr.id}: Teórico={theoretical_delays[i]:.6f}s, Generado={tr.stats.delay:.6f}s")

    def __compute_music_spectrum(self, positions, fs, En, signal_vec, slow_lim, true_azimuth=None, true_slowness=None):
        azimuths = np.linspace(0, 360, 361)
        slowness_range = np.linspace(0.2, 0.3, 500)
        freq = 1.0


        # Cambio crucial en la dirección del steering vector
        #def make_steering(az, slowness):
        #    az_rad = np.deg2rad(az)
        #    direction = np.array([np.cos(az_rad), np.sin(az_rad)])  # Cambio a [cos, sin]
        #    delays = positions @ direction * slowness
        #    return np.exp(2j * np.pi * freq * delays)  # Cambio de signo en la fase

        # Modificar la función make_steering:
        #def make_steering(az, slowness):
        #    az_rad = np.deg2rad(az)
        #    direction = np.array([np.cos(az_rad), np.sin(az_rad)])  # Coordenadas Este/Norte
        #    delays = positions @ direction * slowness
        #    return np.exp(2j * np.pi * freq * delays)  # Signo positivo para coincidir con generación
        def make_steering(az, slowness):
            az_rad = np.deg2rad(az)
            direction = np.array([np.cos(az_rad), np.sin(az_rad)])

            delays = positions @ direction * slowness
            return np.exp(-2j * np.pi * freq * delays)  # Signo negativo para fase consistente

        # Inicializar music_map
        music_map = np.zeros((len(slowness_range), len(azimuths)))

        # Llenar music_map
        for i, s in enumerate(slowness_range):
            for j, az in enumerate(azimuths):
                steering = make_steering(az, s)[:, np.newaxis]
                steering = steering / np.linalg.norm(steering)  # Normalizar
                proj = steering.T.conj() @ En @ En.T.conj() @ steering
                music_map[i, j] = 1 / (np.abs(proj.item()) + 1e-10)  # Usar .item() para evitar warning

        # Encontrar pico
        peak_idx = np.argmax(music_map)
        peak_slowness = slowness_range[peak_idx // len(azimuths)]
        peak_azimuth = azimuths[peak_idx % len(azimuths)]  # Definir peak_azimuth aquí primero

        print(f"Pico inicial en: Azimuth={peak_azimuth:.1f}°, Slowness={peak_slowness:.4f} s/km")


        # Corrección del azimuth estimado
        #peak_azimuth = (peak_azimuth + 180) % 360  # Convertir a BAZ
        # Con:
        #if peak_azimuth >= 180:
        #    peak_baz = peak_azimuth - 180
        #else:
        #    peak_baz = peak_azimuth + 180
        #peak_baz = (peak_azimuth + 180) % 360  # Conversión a BAZ
        peak_baz = peak_azimuth

        print(f"Pico corregido (BAZ): {peak_azimuth:.1f}°")
        if true_azimuth is not None:
            # Ajuste fino para minimizar diferencia angular
            angular_diff = (peak_azimuth - (true_azimuth + 180)) % 360
            if angular_diff > 180:
                angular_diff -= 360
            print(f"Ajuste angular final: {angular_diff:.1f}°")

        # Verificación vectorial mejorada
        if true_azimuth is not None:
            true_az_rad = np.deg2rad(true_azimuth)  # <-- AÑADE ESTA LÍNEA
            true_dir = np.array([np.cos(true_az_rad), np.sin(true_az_rad)])
            true_dir /= np.linalg.norm(true_dir)

            est_az_rad = np.deg2rad(peak_azimuth)  # <-- También convierte el ángulo estimado
            est_dir = np.array([np.cos(est_az_rad), np.sin(est_az_rad)])
            est_dir = est_dir / np.linalg.norm(est_dir)

            print("\n[VALIDACIÓN] Ángulos clave:")
            print(f"Dirección de propagación verdadera: {true_azimuth}°")
            print(f"BAZ estimado: {peak_azimuth}°")
            print(f"Coincidencia direccional: {np.dot(true_dir, est_dir):.6f}")

        # Verificación numérica (versión corregida)
        if true_azimuth is not None:
            true_steering = np.exp(-2j * np.pi * 60 * (positions @
                                                       np.array([np.sin(np.deg2rad(true_azimuth)),
                                                                 np.cos(np.deg2rad(true_azimuth))]) * true_slowness)[:,
                                                      np.newaxis])
            estimated_steering = np.exp(-2j * np.pi * 60 * (positions @
                                                            np.array([np.sin(np.deg2rad(peak_azimuth)),
                                                                      np.cos(
                                                                          np.deg2rad(peak_azimuth))]) * peak_slowness)[
                                                           :, np.newaxis])

            correlation = np.abs(np.vdot(true_steering.flatten(), estimated_steering.flatten()))
            print(f"\n[DEBUG] Correlación post-corrección: {correlation:.6f} (debe ser ~1.0)")
            print(f"BAZ estimado: {peak_azimuth:.1f}° (Verdadero: {(true_azimuth + 180) % 360:.1f}°)")

        # ===== PRUEBA ADICIONAL 2: Comparación de direcciones =====
        if true_azimuth is not None:
            true_dir = np.array([np.sin(np.deg2rad(true_azimuth)), np.cos(np.deg2rad(true_azimuth))])
            est_dir = np.array([np.sin(np.deg2rad(peak_azimuth)), np.cos(np.deg2rad(peak_azimuth))])

            print("\n[DEBUG] Comparación de direcciones:")
            print(f"Vector dirección verdadero: {true_dir}")
            print(f"Vector dirección estimado: {est_dir}")
            print(f"Producto punto (debe ser ~1): {np.dot(true_dir, est_dir):.6f}")

            # ===== PRUEBA ADICIONAL 3: Comparación de fases =====
            theoretical_phase = -2 * np.pi * 60 * (positions @ true_dir * true_slowness)
            estimated_phase = -2 * np.pi * 60 * (positions @ est_dir * peak_slowness)
            print("\n[DEBUG] Diferencia de fases (grados):")
            print(np.rad2deg(theoretical_phase - estimated_phase))

        # ===== PRUEBA ADICIONAL 5: Máscara de calidad =====
        if true_azimuth is not None:
            mask = (music_map > 0.8 * np.max(music_map))
            valid_az = azimuths[mask.any(axis=0)]
            valid_slow = slowness_range[mask.any(axis=1)]
            print(f"\n[DEBUG] Rango válido - Azimuth: {valid_az.min():.1f}° a {valid_az.max():.1f}°")
            print(f"Rango válido - Slowness: {valid_slow.min():.5f} a {valid_slow.max():.5f} s/km")

        # Añadir al final de __compute_music_spectrum():
        if true_azimuth is not None:
            true_baz = (true_azimuth + 180) % 360
            # Cálculo robusto del error angular (elimina falsos 168°)
            #error = min(abs(peak_baz - true_baz), 360 - abs(peak_baz - true_baz))
            #error = min((peak_baz - true_baz) % 360, (true_baz - peak_baz) % 360)
            #error = error if error <= 180 else 360 - error  # Asegura error <= 180°
            error = (peak_baz - true_baz) % 360
            error = error if error <= 180 else 360 - error
            print(f"\n[RESULTADO FINAL] Error angular: {error:.2f}°")
            print(f"Slowness estimada: {peak_slowness:.5f} vs verdadera: {true_slowness:.5f}")

        return music_map, azimuths, slowness_range, peak_slowness, peak_azimuth, peak_baz  # Devolver BAZ, no azimuth #peak_azimuth  # Devolver azimuth, no BAZ

    def run_music_test(self, true_azimuth=45.0, true_slowness_ms=250.0, fs=100, fc=1, slow_lim = 0.3):

        true_slowness = true_slowness_ms / 1000
        print(f"true_slowness_ms: {true_slowness_ms}, true_slowness: {true_slowness:.6e} s/km")
        #self.__create_fixed_array()
        self.__create_test_array()
        self.get_test_traces(fs, fc, true_azimuth, true_slowness_ms)
        data = self.data_array
        sensor_positions = self.__get_sensor_positions()
        fs = self.st[0].stats.sampling_rate
        R = self.__get_covariance_matrix(data)
        En, signal_vec = self.__get_noise_subspace(R, n_signals=1)
        music_map, azimuths, slowness_range, peak_slowness, peak_azimuth, peak_baz  = self.__compute_music_spectrum(
            sensor_positions, fs, En, signal_vec, slow_lim, true_azimuth, true_slowness
        )

        # ===== PRUEBA ADICIONAL 4: Graficar vector propio vs steering =====

        true_steering = np.exp(-2j * np.pi * 60 * (sensor_positions @ np.array(
            [np.sin(np.deg2rad(true_azimuth)), np.cos(np.deg2rad(true_azimuth))]) * true_slowness))

        # plt.figure(figsize=(10, 4))
        # plt.subplot(121)
        # plt.title("Parte Real")
        # plt.plot(signal_vec.real, 'bo-', label='Vector propio')
        # plt.plot(true_steering.real, 'rx--', label='Steering teórico')
        # plt.legend()
        #
        # plt.subplot(122)
        # plt.title("Parte Imaginaria")
        # plt.plot(signal_vec.imag, 'bo-', label='Vector propio')
        # plt.plot(true_steering.imag, 'rx--', label='Steering teórico')
        # plt.legend()
        #
        # plt.tight_layout()
        # plt.show()
        #peak_baz = (peak_azimuth + 180) % 360  # Conversión a BAZ aquí
        self.__plot_slowness_map(music_map, azimuths, slowness_range, peak_slowness, peak_baz, true_azimuth)


if __name__ == "__main__":
    ATT = ArrayToolsTest()
    print("Creando datos de prueba...")
    ATT.run_music_test(true_azimuth=90.0, true_slowness_ms=250.0, fs=100, fc=1)