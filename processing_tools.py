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
from scipy.signal import hilbert
from scipy.linalg import eigh

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
            coords = self.inv.get_coordinates(self.st[i].id, self.st[i].stats.starttime)

            self.st[i].stats.coordinates = AttribDict(
                {'latitude': coords['latitude'], 'elevation': coords['elevation'], 'longitude': coords['longitude']})

            self.coords[self.st[i].id] = [
                self.st[i].stats.coordinates.latitude,
                self.st[i].stats.coordinates.longitude]


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


    def run_music(self, slow_lim = 0.3):
        print("Probando algoritmo MUSIC con datos reales...")

        data, num_sensors, num_samples = self.__get_data_dimensions()
        sensor_positions = self.__get_sensor_positions()
        fs = self.st[0].stats.sampling_rate
        R = self.__get_covariance_matrix(data)
        En = self.__get_noise_subspace(R, num_samples)
        music_map, azimuths, slowness_range, peak_slowness,  peak_baz = self.__compute_music_spectrum(sensor_positions, fs, En, slow_lim)
        self.__plot_slowness_map(music_map, azimuths, slowness_range, peak_slowness,  peak_baz)

    def __plot_slowness_map(self, music_map, theta_math, slowness_range, peak_slowness, peak_baz):

        # Calcular coordenadas polares del pico
        peak_theta_rad = np.deg2rad((90 - theta_math) % 360 + 180)
        music_map = 10*np.log(music_map / np.max(music_map))
        music_map = np.clip(music_map, a_min=-5.0, a_max=0)
        # Plot en coordenadas polares
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, polar=True)

        # Crear grilla polar
        r, theta = np.meshgrid(slowness_range * 1E3, peak_theta_rad, indexing='ij')
        c = ax.contourf(theta, r, music_map, shading='auto', cmap='rainbow', levels=100, vmin = -5.0, vmax=0)

        # Flecha hacia el pico
        arrow_length = peak_slowness  # slowness en ms/km
        ax.annotate('', xy=(np.deg2rad(peak_baz), arrow_length),
                    xytext=(0, 0),
                    arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=1.5),
                    annotation_clip=False)

        # Colorbar y estilo
        plt.colorbar(c, label='Power dB')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Ticks blancos
        ax.tick_params(axis='y', colors='white')  # slowness ticks
        ax.tick_params(axis='x', colors='white')  # opcional: azimuth ticks también

        # Título y leyenda
        ax.set_title("MUSIC Spectrum", color='white')
        ax.legend([f'Peak BAZ: {peak_baz:.1f}°, S: {peak_slowness * 1E3:.2f} ms/km'], loc='upper right')
        ax.grid(color='white', linestyle='--', linewidth=0.5)

        plt.show()

    def __get_data_dimensions(self):
        data = self.data_array
        num_sensors, num_samples = data.shape
        return data, num_sensors, num_samples

    def __get_sensor_positions(self):

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
            x = dist_m * np.cos(np.radians(az_deg))  # Este
            y = dist_m * np.sin(np.radians(az_deg))  # Norte
            sensor_positions.append([x, y])
        return np.array(sensor_positions)

    
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

    def __compute_music_spectrum(self, positions, fs, En, slow_lim, res_angle= "high", res_slow= "high"):

        slowness_lim = slow_lim * 1E-3
        if res_angle == "high":
            azimuths = np.linspace(0, 360, 5*360)
        elif res_slow=="low":
            azimuths = np.linspace(0, 360, 360)


        if res_slow=="high":
            slowness_resolution = 0.025*slowness_lim
        elif res_slow=="low":
            slowness_resolution = 0.05 * slowness_lim

        slowness_range = np.arange(0, slowness_lim, slowness_resolution)
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

        # Obtener pico máximo del MUSIC
        i_max, j_max = np.unravel_index(np.argmax(music_map), music_map.shape)
        peak_slowness = slowness_range[i_max]*1E3
        theta_math = azimuths[j_max]
        peak_baz = (90 - theta_math) % 360 + 180
        print(f"Maximum Baz [º]: {peak_baz:.1f}", f"Maximum Slowness [s/km]: {peak_slowness:.3f}" )
        return music_map, azimuths, slowness_range, peak_slowness,  peak_baz

    def __estimate_peak(self, music_map, azimuths, slowness_range):
        idx = np.unravel_index(np.argmax(music_map), music_map.shape)
        est_az = azimuths[idx[1]]
        est_vel = 1 / slowness_range[idx[0]]
        return est_az, est_vel



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
    start_date = "2017-09-03 03:39:05"
    end_date = "2017-09-03 03:39:08"
    AT.get_traces(starttime = start_date, endtime=end_date)
    AT.run_music()
