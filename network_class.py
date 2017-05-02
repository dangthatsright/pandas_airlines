import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

class Network(object):
    def __init__(self, df):
        self.df = df
        self.ID_set = list(set(self.df['ORIGIN_AIRPORT_ID']) | set(self.df['DEST_AIRPORT_ID']))
        self.adjMatrix = None
        self.initAdjMatrix()
        self.ID_to_Airport = None
        self.ID_dict()
        self.Airport_to_ID = {v: k for k, v in self.ID_to_Airport.iteritems()}

        self.degrees = {} #{airport iD:[in degree, out degree]}
        self.all_degrees()
        
    # Currently, the matrix doesn't specify by airlines
    # also aggregates people over the year 
    # A[i,j] represents the #travelers from i to j over the whole year 
    def initAdjMatrix(self):
        numAirports = len(self.ID_set)        
        self.adjMatrix = np.zeros((numAirports, numAirports))
        
        for origin in self.ID_set:
            for dest in set(self.df[self.df['ORIGIN_AIRPORT_ID']==origin]['DEST_AIRPORT_ID']):
                flights = self.df[(self.df['ORIGIN_AIRPORT_ID']==origin) & (self.df['DEST_AIRPORT_ID']==dest)]
                total_passengers = sum(flights['PASSENGERS'])
                originIDX = self.ID_set.index(origin)
                destIDX = self.ID_set.index(dest)
                self.adjMatrix[originIDX, destIDX] = total_passengers
    
    def ID_dict(self):
        origins = self.df[['ORIGIN_AIRPORT_ID','ORIGIN']].as_matrix()
        self.ID_to_Airport = {origins[i,0]: origins[i,1] for i in xrange(len(origins))}
        dests = self.df[['DEST_AIRPORT_ID','DEST']].as_matrix()
        for i in xrange(len(dests)):
            self.ID_to_Airport[dests[i,0]] = dests[i,1]
    
    #returns set of IDs that are neighbors to the current ID 
    def all_neighbors(self, ID):
        i = self.ID_set.index(ID)
        neighbors = set()
        for j in xrange(len(self.adjMatrix)):
            if (self.adjMatrix[i,j] != 0) or (self.adjMatrix[j,i] != 0):
                neighbors.add(self.ID_set[j])

        return neighbors
    
    #returns set of neighbors leading out of a node
    def out_neighbors(self, ID):
        i = self.ID_set.index(ID)
        neighbors = set()
        for j in xrange(len(self.adjMatrix)):
            if (self.adjMatrix[i,j] != 0):
                neighbors.add(self.ID_set[j])

        return neighbors
    
    #returns set of neighbors leading into a node
    def in_neighbors(self, ID):
        i = self.ID_set.index(ID)
        neighbors = set()
        for j in xrange(len(self.adjMatrix)):
            if (self.adjMatrix[j,i] != 0):
                neighbors.add(self.ID_set[j])

        return neighbors 

    def all_degrees(self):
        for airport in self.ID_set:
            num_in = len(self.in_neighbors(airport))
            num_out = len(self.out_neighbors(airport))
            
            self.degrees[airport] = [num_in, num_out]
            
    #observed degree distribution for some degree d 
    def out_degree_dist(self, d):
        n = len(self.ID_set)
        num = 0
        for key, val in self.degrees.iteritems():
            if val[1] == d: num += 1
        return (1.0*num)/n
    
    def in_degree_dist(self, d):
        n = len(self.ID_set)
        num = 0
        for key, val in self.degrees.iteritems():
            if val[0] == d: num += 1
        return (1.0*num)/n
    
    def plot_degree_dist(self, start, end):
        x = range(start,end)
        y1 = [flightNetwork.out_degree_dist(i) for i in x]
        y2 = [flightNetwork.in_degree_dist(i) for i in x]
        
        plt.plot(x,y1)
        plt.plot(x,y2)
        plt.show()
  