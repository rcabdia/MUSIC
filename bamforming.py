#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bamforming
"""
import math
from datetime import datetime
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import hilbert
from scipy.linalg import eigh
from obspy import Trace, Stream, UTCDateTime, read_inventory
from obspy.core import AttribDict
from random import uniform
import os
from obspy import read


class BeamForming:
    def __init__(self, path_files, path_metadata):
        self.inv = None  # Inventario de metadatos
        self.st = None  # Flujo de datos sísmicos
        self.path_files = path_files  # Ruta a los archivos de datos sísmicos
        self.path_metadata = path_metadata  # Ruta a los archivos de metadatos

    def __get_inventory(self):
        self.inv = read_inventory(self.path_metadata)
    def get_traces(self, starttime=None, endtime=None):
        self.__get_inventory()
        self.__read_files(starttime, endtime)
        self.__extract_coordinates()
        self.__convert_to_array()
    def __get_sensor_positions(self):
        sensor_positions = []
        for tr_id in self.index_list:
            y, x = self.relative_coords_km[tr_id]
            sensor_positions.append([x, y])
            print(f"Sensor {tr_id}: Este={x:.3f} km, Norte={y:.3f} km")
        return np.array(sensor_positions)

    def __read_files(self, starttime, endtime):
        date_format = "%Y-%m-%d %H:%M:%S"
        if starttime is not None and endtime is not None:
            start = datetime.strptime(starttime, date_format)
            start = UTCDateTime(start)
            end = datetime.strptime(endtime, date_format)
            end = UTCDateTime(end)

        traces = []
        for root, dirs, files in os.walk(self.path_files):
            # Nombre de los archivos: NET.STA.LOC.CHAN.DYYYY.DDD
            for file in files:
                #print(file)
                try:
                    #print(len(read(os.path.join(root, file))))
                    tr = read(os.path.join(root, file))[0]  #[0] extrae el primer "Trace" de ese Stream, que representa una señal individual (comp. de una estación, por ejemplo).
                    #tr.filter(type="bandpass", freqmin=1, freqmax=3, zerophase=True)
                    if starttime is not None and endtime is not None:
                       tr.trim(start, end)
                       tr.detrend(type="simple")
                    traces.append(tr)  #Añade la traza (tr) a una lista llamada traces, que acumula todas las señales válidas para trabajar con ellas después.
                except:
                    print("file:", file, "Not valid")

        self.st = Stream(traces)

    def __extract_coordinates(self):
        R = 6371.0  # Earth radius in km
        n = len(self.st)
        self.coords = {}
        lats = []
        lons = []

        for i in range(n):
            coords = self.inv.get_coordinates(self.st[i].id, self.st[i].stats.starttime)
            lat = coords['latitude']
            lon = coords['longitude']

            self.st[i].stats.coordinates = AttribDict({
                'latitude': lat,
                'elevation': coords['elevation'],
                'longitude': lon
            })

            self.coords[self.st[i].id] = [lat, lon]
            lats.append(lat)
            lons.append(lon)

        # Compute barycenter
        barycenter_lat = np.mean(lats)
        barycenter_lon = np.mean(lons)

        # Convert lat/lon to relative x, y in km
        self.relative_coords_km = {}
        for key, (lat, lon) in self.coords.items():
            delta_lat = math.radians(lat - barycenter_lat)
            delta_lon = math.radians(lon - barycenter_lon)
            mean_lat_rad = math.radians((lat + barycenter_lat) / 2.0)

            dx = R * delta_lon * math.cos(mean_lat_rad)  # East-West (x)
            dy = R * delta_lat  # North-South (y)

            self.relative_coords_km[key] = (dx, dy)

    def __convert_to_array(self):
        n = len(self.st)
        if n == 0:
            raise ValueError("No hay trazas para convertir")
        npts = self.st[0].stats.npts
        self.data_array = np.zeros((n, npts), dtype=complex)
        self.index_list = []
        for i, tr in enumerate(self.st):
            if len(tr.data) != npts:
                raise ValueError(f"Todas las trazas deben tener igual longitud. Traza {tr.id} tiene {len(tr.data)} muestras")
            self.data_array[i, :] = hilbert(tr.data)
            self.index_list.append(tr.id)
        print(f"Array convertido: {self.data_array.shape} (sensores x muestras)")

    def get_traces(self, starttime=None, endtime=None):
        self.__get_inventory()
        self.__read_files(starttime, endtime)
        self.__extract_coordinates()
        self.__convert_to_array()
    def __get_covariance_matrix(self):
        if self.data_array.ndim != 2:
            raise ValueError(f"Se esperaba array 2D, se recibió {self.data_array.ndim}D")
        R = np.dot(self.data_array, self.data_array.T.conj()) / self.data_array.shape[1]
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

    def __make_steering(self, positions, az, slowness, freq):
        az_rad = np.deg2rad(az)
        direction = np.array([np.cos(az_rad), np.sin(az_rad)])
        delays = positions @ direction * slowness
        return np.exp(-2j * np.pi * freq * delays)

    def compute_music_spectrum(self, positions, fs, En, signal_vec, slow_lim):

        azimuths = np.linspace(0, 360, 361)
        slowness_range = np.linspace(0.00, slow_lim, 300)
        freq = 1.5
        music_map = np.zeros((len(slowness_range), len(azimuths)))

        # Calcular mapa MUSIC completo
        for i, s in enumerate(slowness_range):
            for j, az in enumerate(azimuths):
                steering = self.__make_steering(positions, az, s, freq)[:, np.newaxis]
                steering = steering / np.linalg.norm(steering)
                proj = steering.T.conj() @ En @ En.T.conj() @ steering
                music_map[i, j] = 1 / (np.abs(proj.item()) + 1e-10)

        # Encontrar los picos

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

    def __plot_array_geometry(self, max_aperture_km=None, show_labels=True):
        plt.figure(figsize=(8, 8))
        for sensor_id, (x, y) in self.relative_coords_km.items():
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
    def run_music(self, slow_lim=0.4, n_signals=1):

        fs = self.st[0].stats.sampling_rate

        self.st.filter(type="bandpass", freqmin=0.5, freqmax=2.0, zerophase=True, corners=4)
        self.st.plot()
        R = self.__get_covariance_matrix()
        sensor_positions = self.__get_sensor_positions()
        self.__plot_array_geometry()
        # Determinar número de señales a buscar
        #seonsor_positions =
        En, signal_vec = self.__get_noise_subspace(R, n_signals=n_signals)

        # Recibir los 8 valores devueltos
        (music_map1, music_map2, azimuths, slowness_range,
         peak_slowness1, peak_baz1, peak_slowness2, peak_baz2) = self.compute_music_spectrum(
            sensor_positions, fs, En, signal_vec, slow_lim)
        self.__plot_slowness_map(music_map1, azimuths, slowness_range, peak_slowness1, peak_baz1)
        #
        # self.__plot_slowness_map(music_map1, azimuths, slowness_range,
        #                        peak_slowness1, peak_baz1, peak_slowness2, peak_baz2, true_azimuth1, true_slowness1,
        #                        true_azimuth2, true_slowness2 if true_azimuth2 is not None else None)

    def __plot_slowness_map(self, music_map, azimuths, slowness_range,
                            peak_slowness1, peak_baz1):

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
            ax1.plot(np.deg2rad(peak_baz1), peak_slowness1, 'w*', markersize=5,
                     label=f'Pico 1: {peak_baz1:.1f}°, {peak_slowness1:.3f} s/km')

        # Gráfico de información de parámetros
        ax2 = plt.subplot(122)
        ax2.set_title("Estimación de Parámetros", pad=20)
        ax2.axis('off')

        info_text = ""
        if peak_baz1 is not None:
            info_text += f"Fuente 1 estimada:\n"
            info_text += f"- Backazimuth: {peak_baz1:.1f}°\n"
            info_text += f"- Slowness: {peak_slowness1:.3f} s/km\n"
            info_text += "\n"


        # Imprimir por consola
        print("\nRESULTADOS DE ESTIMACIÓN:")
        print(info_text)

        ax2.text(0.1, 0.5, info_text, fontsize=12, va='center')
        ax1.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    """
    path_files = "/Users/admin/Documents/iMacROA/MUSIC/data_cut"
    metadata_path = "/Users/admin/Documents/iMacROA/MUSIC/metadata/dataless.dlsv"
    """

    # Obtener la ruta del directorio donde está el script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_files = os.path.join(script_dir, "data_cut")
    metadata_path = os.path.join(script_dir, "metadata/dataless.dlsv")

    AT = BeamForming(path_files, metadata_path)
    # date_format = "%Y-%m-%d %H:%M:%S"  # "2023-12-11 14:30:00"
    start_date = "2017-09-03 03:39:05"
    end_date = "2017-09-03 03:39:08"
    AT.get_traces(starttime=start_date, endtime=end_date)
    AT.run_music(slow_lim=0.2)

