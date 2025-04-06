#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
processing_tools_array
"""
import os
from datetime import datetime
import numpy as np
from obspy import read, Inventory, Stream, read_inventory, UTCDateTime
from obspy.core import AttribDict
from obspy.core.inventory import Station, Network
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import hilbert
from matplotlib.colors import Normalize
from obspy.geodetics import gps2dist_azimuth
from scipy.linalg import eigh
from pyproj import Proj
from obspy.geodetics import gps2dist_azimuth

class ArrayTools:

    def __init__(self, path_files, path_metadata):
        self.inv = None # Inventario de metadatos 
        self.st = None  # Flujo de datos sísmicos
        self.path_files = path_files  # Ruta a los archivos de datos sísmicos
        self.path_metadata = path_metadata # Ruta a los archivos de metadatos

    def __get_inventory(self):  
        self.inv = read_inventory(self.path_metadata)

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
            #print("-1-")
            #print(root)
            #print(files)
            #print("-2-")
            for file in files:
                #print(file)
                try:
                    #print(len(read(os.path.join(root, file))))
                    tr = read(os.path.join(root, file))[0]  #[0] extrae el primer "Trace" de ese Stream, que representa una señal individual (comp. de una estación, por ejemplo).
                    if starttime is not None and endtime is not None:
                       tr.trim(start, end)
                    traces.append(tr)  #Añade la traza (tr) a una lista llamada traces, que acumula todas las señales válidas para trabajar con ellas después.
                except:
                    print("file:", file, "Not valid")

        #for tr in traces:
        #    print(f"Tasa de muestreo de {tr.id}: {tr.stats.sampling_rate} Hz")
        
        self.st = Stream(traces)

    def __filter_inventory_by_stream(self, stream: Stream, inventory: Inventory) -> Inventory:
        # Create an empty list to hold filtered networks
        filtered_networks = []

        # Loop through networks in the inventory
        for network in inventory:
            # Create a list to hold filtered stations for each network
            filtered_stations = []

            # Loop through stations in the network
            for station in network:
                # Find channels in this station that match the stream traces
                filtered_channels = []

                # Check if any trace in the stream matches the station and network
                for trace in stream:
                    # Extract network, station, location, and channel codes from trace
                    trace_net, trace_sta, trace_loc, trace_chan = trace.id.split(".")

                    # Check if the current station and network match the trace
                    if station.code == trace_sta and network.code == trace_net:
                        # Look for a channel in the station that matches the trace's channel code
                        for channel in station.channels:
                            if channel.code == trace_chan and (not trace_loc or channel.location_code == trace_loc):
                                filtered_channels.append(channel)

                # If there are any matching channels, create a filtered station
                if filtered_channels:
                    filtered_station = Station(
                        code=station.code,
                        latitude=station.latitude,
                        longitude=station.longitude,
                        elevation=station.elevation,
                        creation_date=station.creation_date,
                        site=station.site,
                        channels=filtered_channels
                    )
                    filtered_stations.append(filtered_station)

            # If there are any matching stations, create a filtered network
            if filtered_stations:
                filtered_network = Network(
                    code=network.code,
                    stations=filtered_stations
                )
                filtered_networks.append(filtered_network)

        # Create a new inventory with the filtered networks
        filtered_inventory = Inventory(networks=filtered_networks, source=inventory.source)
        return filtered_inventory
        
    def __extract_coordinates(self):
        n = len(self.st)
        #print(f"valor de n:{n}")
        self.coords = {} # Diccionario vacío donde se almacenarán las coordenadas de cada traza, que es una traza de una estación
        for i in range(n):
            #coords = self.inv.get_coordinates(self.st[i].id)
            #se usa starttime para evitar ambigüedades si hay varios registro históricos, y el warning 
            #print("-1-")
            #print(self.inv)
            #print(self.st[i].id)
            coords = self.inv.get_coordinates(self.st[i].id, self.st[i].stats.starttime)
            #print(coords)
            #print("-2-")
            self.st[i].stats.coordinates = AttribDict(
                {'latitude': coords['latitude'], 'elevation': coords['elevation'], 'longitude': coords['longitude']})
            #print("coordinates:", self.st[i].stats.coordinates.latitude,  self.st[i].stats.coordinates.longitude)
            self.coords[self.st[i].id] = [
                self.st[i].stats.coordinates.latitude,
                self.st[i].stats.coordinates.longitude
            ]
        #print(self.coords)

    def __convert_to_array(self):
        # fila: traza sísmica
        # columna: valor de la señal en el tiempo
        # Lo siguiente asume que todas las trazas tienen la misma longitud, lo cual es común si se ha hecho un trim() antes
        self.data_array = np.zeros((len(self.st), len(self.st[0].data))) 
        self.index_list = []
        for i, tr in enumerate(self.st):
            #self.index_list.append(self.st[i].id)
            #self.data_array[i,:] = self.st[i].data
            self.index_list.append(tr.id)
            self.data_array[i,:] = tr.data            
        print("end __convert_to_array")

    def get_traces(self, starttime=None, endtime=None):
        self.__get_inventory()
        self.__read_files(starttime, endtime)
        self.__extract_coordinates()
        self.__convert_to_array()        

        # === 1. Subplots para cada traza ===
        num_traces = len(self.st)
        fig, axs = plt.subplots(num_traces, 1, figsize=(10, 2 * num_traces), sharex=True)
        #fig, axs = plt.subplots(num_traces, 1, figsize=(14, 2 * num_traces), sharex=True)

        for i, tr in enumerate(self.st):
            t = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts)
            axs[i].plot(t, tr.data)
            #axs[i].set_ylabel(tr.id, fontsize=8)
            station_code = tr.id.split(".")[1]  # Extrae solo STA
            axs[i].set_ylabel(station_code, fontsize=9, rotation=0, labelpad=40)
            axs[i].grid(True)

        axs[-1].set_xlabel("Tiempo (s)")
        plt.suptitle("Trazas sísmicas", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        # === 2. Mapa de ubicaciones ===
        plt.figure(figsize=(7, 7))
        for trace_id, (lat, lon) in self.coords.items():
            #plt.plot(lon, lat, 'o', label=trace_id, markersize=6)  
            plt.plot(lon, lat, 'o', markersize=6)  # punto negro
            plt.text(lon + 0.001, lat + 0.001, trace_id, fontsize=9, ha='left', va='bottom')
        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.title("Ubicación de las estaciones")
        plt.grid(True)
        #plt.legend(fontsize="small", loc="best")        
        plt.tight_layout()
        plt.show()        
        print("end get_traces")

    def run_music(self):
        print("Probando algoritmo MUSIC con datos reales...")

        data, num_sensors, num_samples = self.__get_data_dimensions()
        sensor_positions = self.__get_sensor_positions()
        fs = self.st[0].stats.sampling_rate
        R = self.__get_covariance_matrix(data)
        En = self.__get_noise_subspace(R, num_samples)
        music_map, azimuths, slowness_range = self.__compute_music_spectrum(sensor_positions, fs, En)
        estimated_az, estimated_velocity = self.__estimate_peak(music_map, azimuths, slowness_range)
        #self.__plot_music_spectrum(music_map, azimuths, slowness_range, estimated_az, estimated_velocity)
        self.__plot_slices(music_map, azimuths, slowness_range, estimated_az, estimated_velocity)
        #self.__plot_zoom_spectrum(music_map, azimuths, slowness_range, estimated_az, estimated_velocity)
        print(f"Dirección estimada (azimuth): {estimated_az:.2f}°")
        print(f"Velocidad estimada: {estimated_velocity:.2f} m/s")

    def __get_data_dimensions(self):
        data = self.data_array
        num_sensors, num_samples = data.shape
        return data, num_sensors, num_samples

    def __get_sensor_positions(self):
        """
        # Aproximación simplificada para convertir coordenadas geográficas (lat/lon) a coordenadas cartesianas en metros (x/y).
        ref_lat, ref_lon = list(self.coords.values())[0]  # estación de referencia
        sensor_positions = []
        for tr_id in self.index_list:
            lat, lon = self.coords[tr_id]
            x = (lon - ref_lon) * 111320 * np.cos(np.radians(ref_lat))  # metros aprox.
            y = (lat - ref_lat) * 110540
            sensor_positions.append([x, y])
        sensor_positions = np.array(sensor_positions)
        return sensor_positions
        """
        """
        ref_lat, ref_lon = list(self.coords.values())[0]
        # Definir proyección UTM local (unidades en metros)
        utm_proj = Proj(proj="utm", zone=self.__get_utm_zone(ref_lon), ellps="WGS84")
        sensor_positions = []
        for tr_id in self.index_list:
            lat, lon = self.coords[tr_id]
            x, y = utm_proj(lon, lat)  # Conversión precisa a metros
            sensor_positions.append([x, y])
        return np.array(sensor_positions)
        """
        ref_lat, ref_lon = list(self.coords.values())[0]
        sensor_positions = []
        for tr_id in self.index_list:
            lat, lon = self.coords[tr_id]
            # Calcular distancia y azimuth respecto a la referencia
            dist_m, az_deg, _ = gps2dist_azimuth(ref_lat, ref_lon, lat, lon)
            # Convertir a coordenadas locales (Este, Norte)
            x = dist_m * np.sin(np.radians(az_deg))  # Este
            y = dist_m * np.cos(np.radians(az_deg))  # Norte
            sensor_positions.append([x, y])
        return np.array(sensor_positions)
        
    """
    def __get_utm_zone(self, longitude):
        return int((longitude + 180) // 6) + 1  # Cálculo automático de zona UTM
    """
    
    def __get_covariance_matrix(self, data):
        analytic = hilbert(data)
        # Opción 1: Normalizada por N
        R = (analytic @ analytic.conj().T) / data.shape[1]
        # Opción 2: Sin normalizar (para señales débiles)
        #R = analytic @ analytic.conj().T
        return R

    def __get_noise_subspace(self, R, num_samples):
        eigvals, eigvecs = eigh(R)
        idx = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:, idx]
        return eigvecs[:, 1:]  # Asumiendo una sola fuente

    def __compute_music_spectrum(self, positions, fs, En):
        c = 343.0  # velocidad referencia
        
        #azimuths = np.linspace(0, 360, 360)
        azimuths = np.linspace(0, 360, 720) # mejor resolucion, 0.5°

        #slowness_range = np.linspace(1/1500, 1/500, 200)
        #slowness_range = np.linspace(1/6000, 1/300, 200)  # 300–6000 m/s. slowness_range=1/6000 a 1/300: Captura tanto ondas P (alta velocidad) como S (baja velocidad).        
        slowness_range = np.linspace(1/5000, 1/800, 200)

        music_map = np.zeros((len(slowness_range), len(azimuths)))

        for i, s in enumerate(slowness_range):
            for j, az in enumerate(azimuths):
                az_rad = np.deg2rad(az)
                direction = np.array([np.cos(az_rad), np.sin(az_rad)])
                delays = positions @ direction * s
                steering = np.exp(-2j * np.pi * fs * delays[:, np.newaxis])
                proj = steering.conj().T @ En @ En.conj().T @ steering
                power = np.abs(proj).sum()  #.sum suaviza el espectro
                music_map[i, j] = 1 / power
        return music_map, azimuths, slowness_range

    def __estimate_peak(self, music_map, azimuths, slowness_range):
        idx = np.unravel_index(np.argmax(music_map), music_map.shape)
        est_az = azimuths[idx[1]]
        est_vel = 1 / slowness_range[idx[0]]
        return est_az, est_vel

    def __plot_music_spectrum(self, music_map, azimuths, slowness_range, est_az, est_vel):

        """
        music_db = 10 * np.log10(music_map)
        vmin = np.percentile(music_db, 80)
        vmax = np.max(music_db)
        norm = Normalize(vmin=vmin, vmax=vmax)

        plt.figure(figsize=(12, 6))
        plt.imshow(
            music_db,
            extent=[azimuths[0], azimuths[-1], 1/slowness_range[-1], 1/slowness_range[0]],
            aspect='auto', origin='lower', cmap='hot', norm=norm
        )
        plt.colorbar(label='Nivel (dB)')
        plt.title("MUSIC spectrum: dirección vs velocidad")
        plt.xlabel("Azimuth (\u00b0)")
        plt.ylabel("Velocidad (m/s)")
        plt.tight_layout()
        plt.show()
        """
        plt.figure(figsize=(12, 6))
        im = plt.imshow(
            10 * np.log10(music_map),
            extent=[0, 360, 1 / slowness_range[-1], 1 / slowness_range[0]],
            aspect='auto',
            origin='lower',
            cmap='plasma'
        )
        plt.colorbar(label='Nivel (dB)')
        plt.title("MUSIC spectrum: dirección vs velocidad", fontsize=13)
        plt.xlabel("Azimuth (°)")
        plt.ylabel("Velocidad (m/s)")

        # Marcar máximo con símbolo visible
        plt.scatter(
            est_az, est_vel,
            s=150, marker='o',
            facecolors='none',  # transparente por dentro
            edgecolors='red',   # contorno visible
            linewidths=2,
            label='Máximo estimado'
        )

        # Añadir texto con flecha
        plt.annotate(
            f"{est_az:.1f}°, {est_vel:.0f} m/s",
            xy=(est_az, est_vel),
            xytext=(est_az + 10, est_vel + 100),
            arrowprops=dict(arrowstyle="->", color="white", lw=1),
            fontsize=10,
            color="white"
        )

        plt.legend(loc='upper right', fontsize=9, facecolor='white')
        plt.tight_layout()
        plt.show() 
        
    def __plot_slices(self, music_map, azimuths, slowness_range, est_az, est_vel):

        vel_axis = 1 / slowness_range
        az_idx = np.argmin(np.abs(azimuths - est_az))
        vel_idx = np.argmin(np.abs(vel_axis - est_vel))
        az_profile = music_map[vel_idx, :]
        vel_profile = music_map[:, az_idx]

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Dirección
        axs[0].plot(azimuths, 10 * np.log10(az_profile), color='blue')
        peak_az_idx = np.argmax(az_profile)
        peak_az = azimuths[peak_az_idx]
        peak_az_val = 10 * np.log10(az_profile[peak_az_idx])
        axs[0].scatter(peak_az, peak_az_val, color='red', marker='x', s=100, label=f"{peak_az:.1f}°")
        axs[0].annotate(f"{peak_az_val:.1f} dB", (peak_az, peak_az_val), textcoords="offset points", xytext=(10,10), ha='center')
        axs[0].set_xlabel("Azimuth (°)")
        axs[0].set_ylabel("Nivel (dB)")
        axs[0].set_title("Dirección estimada")
        axs[0].legend()

        # Velocidad
        axs[1].plot(vel_axis, 10 * np.log10(vel_profile), color='green')
        peak_vel_idx = np.argmax(vel_profile)
        peak_vel = vel_axis[peak_vel_idx]
        peak_vel_val = 10 * np.log10(vel_profile[peak_vel_idx])
        axs[1].scatter(peak_vel, peak_vel_val, color='red', marker='x', s=100, label=f"{peak_vel:.0f} m/s")
        axs[1].annotate(f"{peak_vel_val:.1f} dB", (peak_vel, peak_vel_val), textcoords="offset points", xytext=(10,10), ha='center')
        axs[1].set_xlabel("Velocidad (m/s)")
        axs[1].set_ylabel("Nivel (dB)")
        axs[1].set_title("Velocidad estimada")
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def __plot_zoom_spectrum(self, music_map, azimuths, slowness_range, est_az, est_vel):

        az_range = 30
        vel_range = 500
        az_min = max(0, est_az - az_range)
        az_max = min(360, est_az + az_range)
        vel_min = max(1 / slowness_range[-1], est_vel - vel_range)
        vel_max = min(1 / slowness_range[0], est_vel + vel_range)

        az_mask = (azimuths >= az_min) & (azimuths <= az_max)
        vel_axis = 1 / slowness_range
        vel_mask = (vel_axis >= vel_min) & (vel_axis <= vel_max)

        az_subset = azimuths[az_mask]
        vel_subset = vel_axis[vel_mask]
        music_subset = music_map[np.ix_(vel_mask, az_mask)]
        music_db_subset = 10 * np.log10(music_subset)

        vmin = np.percentile(music_db_subset, 80)
        vmax = np.max(music_db_subset)
        norm = Normalize(vmin=vmin, vmax=vmax)

        highlight_mask = music_db_subset >= (vmax - 3)

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            music_db_subset,
            extent=[az_min, az_max, vel_min, vel_max],
            aspect='auto', origin='lower', cmap='hot', norm=norm
        )
        ax.contour(
            az_subset, vel_subset, highlight_mask,
            levels=[0.5], colors='cyan', linewidths=1.2
        )
        plt.colorbar(im, label='Nivel (dB)')
        ax.set_title("MUSIC spectrum (zoom en máximos)")
        ax.set_xlabel("Azimuth (\u00b0)")
        ax.set_ylabel("Velocidad (m/s)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    """
    path_files = "/Users/admin/Documents/iMacROA/MUSIC/data_cut"
    metadata_path = "/Users/admin/Documents/iMacROA/MUSIC/metadata/dataless.dlsv"
    """
    
    # Obtener la ruta del directorio donde está el script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_files = os.path.join(script_dir,"data_cut")
    metadata_path = os.path.join(script_dir,"metadata/dataless.dlsv")
    
    AT = ArrayTools(path_files, metadata_path)
    #date_format = "%Y-%m-%d %H:%M:%S"  # "2023-12-11 14:30:00"
    start_date = "2017-09-03 03:39:04"
    end_date = "2017-09-03 03:39:08"
    AT.get_traces(starttime = start_date, endtime=end_date)
    AT.run_music()
