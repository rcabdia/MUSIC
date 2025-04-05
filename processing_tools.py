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
            #print(dirs)
            #print(files)
            #print("-2-")
            for file in files:
                print(file)
                try:
                    #print(len(read(os.path.join(root, file))))
                    tr = read(os.path.join(root, file))[0]  #[0] extrae el primer "Trace" de ese Stream, que representa una señal individual (comp. de una estación, por ejemplo).
                    if starttime is not None and endtime is not None:
                       tr.trim(start, end)
                       print("++")
                    traces.append(tr)  #Añade la traza (tr) a una lista llamada traces, que acumula todas las señales válidas para trabajar con ellas después.
                except:
                    print("file:", file, "Not valid")

        tr.plot()
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
        print(f"valor de n:{n}")
        self.coords = {}
        for i in range(n):
            #coords = self.inv.get_coordinates(self.st[i].id)
            coords = self.inv.get_coordinates(self.st[i].id, self.st[i].stats.starttime)

            self.st[i].stats.coordinates = AttribDict(
                {'latitude': coords['latitude'], 'elevation': coords['elevation'], 'longitude': coords['longitude']})
            #print("coordinates:", self.st[0].stats.coordinates.latitude,  self.st[0].stats.coordinates.longitude)
            print("coordinates:", self.st[i].stats.coordinates.latitude,  self.st[i].stats.coordinates.longitude)

            #self.coords[self.st[i].id] = [self.st[0].stats.coordinates.latitude,  self.st[0].stats.coordinates.longitude]
            self.coords[self.st[i].id] = [
                self.st[i].stats.coordinates.latitude,
                self.st[i].stats.coordinates.longitude
            ]

    def __convert_to_array(self):
        self.data_array = np.zeros((len(self.st), len(self.st[0].data)))
        self.index_list = []
        for i, tr in enumerate(self.st):
            self.index_list.append(self.st[i].id)
            self.data_array[i,:] = self.st[i].data
        print("end __convert_to_array")

    def get_traces(self, starttime=None, endtime=None):
        self.__get_inventory()
        self.__read_files(starttime, endtime)
        self.__extract_coordinates()
        self.__convert_to_array()
        print("end get_traces")

    def run_music(self):
        i = 0
        for index, tr in zip(self.index_list, self.st):
            coords = self.coords[index]
            # also coords = tr.stats.coordinates.latitude
            data_trace = self.data_array[i,:]
            # also data_trace = tr.data
            i = i+1
        print("done run_music")

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
