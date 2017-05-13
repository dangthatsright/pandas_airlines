# %load C:\\Users\\Richard\\workspace\\pandas_airlines\\network_class.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
import heapq

class Network(object):
    def __init__(self, df):
        self.df = df
        self.ID_set = list(set(self.df['ORIGIN_AIRPORT_ID']) | set(self.df['DEST_AIRPORT_ID']))
        self.adjMatrix = None
        self.initAdjMatrix()
        self.ID_to_Airport = None
        self.ID_dict()
        self.Airport_to_ID = {v: k for k, v in self.ID_to_Airport.items()}
        
        self.degrees = {} #{airport iD:[in degree, out degree]}
        self.all_degrees()
        
        self.fit = None
        #self.fit_regression(1, 101, direction="out") #can change params here
        
    # Currently, the matrix doesn't specify by airlines
    # also aggregates people over the year 
    # A[i,j] represents the #travelers from i to j over the whole year 
    def initAdjMatrix(self):
        numAirports = len(self.ID_set)        
        self.adjMatrix = np.zeros((numAirports, numAirports))
        
        for origin in self.ID_set:
            origin_flights = self.df[self.df['ORIGIN_AIRPORT_ID']==origin]
            for dest in set(origin_flights['DEST_AIRPORT_ID']):
                #flights = self.df[(self.df['ORIGIN_AIRPORT_ID']==origin) & (self.df['DEST_AIRPORT_ID']==dest)]
                flights = origin_flights[origin_flights['DEST_AIRPORT_ID']==dest]
                if 'PASSENGERS' in list(self.df.columns.values):
                    total_passengers = sum(flights['PASSENGERS'])
                elif 'DELAY' in list(self.df.columns.values):
                    total_passengers = sum(flights['DELAY'])
                originIDX = self.ID_set.index(origin)
                destIDX = self.ID_set.index(dest)
                self.adjMatrix[originIDX, destIDX] = total_passengers
    
    def ID_dict(self):
        if 'ORIGIN' in list(self.df.columns.values):
            origins = self.df[['ORIGIN_AIRPORT_ID','ORIGIN']].as_matrix()
        else:
            origins = self.df[['ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_ID']].as_matrix()
        self.ID_to_Airport = {origins[i,0]: origins[i,1] for i in range(len(origins))}
        if 'DEST' in list(self.df.columns.values):
            dests = self.df[['DEST_AIRPORT_ID','DEST']].as_matrix()
        else:
            dests = self.df[['DEST_AIRPORT_ID','DEST_AIRPORT_ID']].as_matrix()
        
        for i in range(len(dests)):
            self.ID_to_Airport[dests[i,0]] = dests[i,1]
    
    #returns set of IDs that are neighbors to the current ID 
    def all_neighbors(self, ID):
        i = self.ID_set.index(ID)
        neighbors = set()
        for j in range(len(self.adjMatrix)):
            if (self.adjMatrix[i,j] != 0) or (self.adjMatrix[j,i] != 0):
                neighbors.add(self.ID_set[j])

        return neighbors
    
    #returns set of neighbors leading out of a node
    def out_neighbors(self, ID):
        i = self.ID_set.index(ID)
        neighbors = set()
        for j in range(len(self.adjMatrix)):
            if (self.adjMatrix[i,j] != 0):
                neighbors.add(self.ID_set[j])

        return neighbors
    
    #returns set of neighbors leading into a node
    def in_neighbors(self, ID):
        i = self.ID_set.index(ID)
        neighbors = set()
        for j in range(len(self.adjMatrix)):
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
        for key, val in self.degrees.items():
            if val[1] == d: num += 1
        return (1.0*num)/n
    
    def in_degree_dist(self, d):
        n = len(self.ID_set)
        num = 0
        for key, val in self.degrees.items():
            if val[0] == d: num += 1
        return (1.0*num)/n
    
    def plot_degree_dist(self, start, end):
        x = range(start,end)
        y1 = [self.out_degree_dist(i) for i in x]
        y2 = [self.in_degree_dist(i) for i in x]
        
        plt.plot(x,y1,label="out-degrees")
        plt.plot(x,y2, label="in-degrees")
        plt.legend(loc = "upper right")
        plt.title("Observed Degree Distribution")
        plt.xlabel("Degrees")
        plt.ylabel("Probability")
        plt.show()
        
    def test_directed(self):
        (directed,undirected) = (0,0)
        for i in range(len(self.adjMatrix)):
            for j in range(len(self.adjMatrix)):
                if (self.adjMatrix[i,j] != 0) and (self.adjMatrix[j,i] != 0):
                    undirected += 1
                elif (self.adjMatrix[i,j] != 0) and (self.adjMatrix[j,i] == 0):
                    directed += 1
                elif (self.adjMatrix[i,j] == 0) and (self.adjMatrix[j,i] != 0):
                    directed += 1
        
        return (undirected/2.0,directed)
    
    def fit_regression(self, start ,end, direction="out"): #direction can be "in" or "out"
        x = range(start,end)
        
        if direction=="out":
            y = [self.out_degree_dist(i) for i in x]
        else:
            y = [self.in_degree_dist(i) for i in x]
        
        cleaned_y = [[x[i], y[i]] for i in range(len(x)) if y[i]!=0.0]
        newx = [i[0] for i in cleaned_y]
        newy = [i[1] for i in cleaned_y]

        regr = linear_model.LinearRegression()
        reshaped_x = np.log(newx)
        reshaped_x = reshaped_x.reshape((len(newx),1))
        regr.fit(reshaped_x, np.log(newy))
        
        print('Coefficients: ', regr.coef_, regr.intercept_)
        print("Mean squared error: %.2f" % np.mean((regr.predict(reshaped_x)-np.log(newy))**2))
        
        return regr
    
    def plot_power_fit(self, start, end, direction="out"):
        
        regr = self.fit_regression(start,end,direction=direction)
        
        x = range(start,end)
        
        y_out = [self.out_degree_dist(i) for i in x]
        y_in = [self.in_degree_dist(i) for i in x]
        
        if direction=="out":
            cleaned_y = [[x[i], y_out[i]] for i in range(len(x)) if y_out[i]!=0.0]
        else:
            cleaned_y = [[x[i], y_in[i]] for i in range(len(x)) if y_in[i]!=0.0]
            
        newx = [i[0] for i in cleaned_y]
                
        plt.plot(x,y_out,label="out-degrees")
        plt.plot(x,y_in, label="in-degrees")
        plt.plot(newx, [np.exp(regr.intercept_)*i**(regr.coef_[0]) for i in newx], label="fit")
        plt.legend(loc = "upper right")
        plt.xlabel("Degrees")
        plt.ylabel("Probability")
        plt.title("Power Fit using "+direction+"-degrees")
        plt.show()
    
    
    def plot_loglog(self, start, end, direction="out"):
        regr = self.fit_regression(start,end,direction=direction)
        
        x = range(start,end)
        
        y_out = [self.out_degree_dist(i) for i in x]
        y_in = [self.in_degree_dist(i) for i in x]
        
        if direction=="out":
            cleaned_y = [[x[i], y_out[i]] for i in range(len(x)) if y_out[i]!=0.0]
        else:
            cleaned_y = [[x[i], y_in[i]] for i in range(len(x)) if y_in[i]!=0.0]
            
        newx = [i[0] for i in cleaned_y]
        reshaped_x = np.log(newx)
        reshaped_x = reshaped_x.reshape((len(newx),1))

        #plt.plot(newx,newy1)
        # plt.plot(np.log(newx), np.log(newy1))
        # plt.plot(np.log(newx), regr.predict(reshaped))
        
        plt.plot(np.log(x),np.log(y_out),label="log out-degrees")
        plt.plot(np.log(x),np.log(y_in), label="log in-degrees")
        plt.plot(np.log(newx), regr.predict(reshaped_x), label="fit")
        plt.legend(loc = "upper right")
        plt.title("Log-Log Graph of Degree Distribution")
        plt.xlabel("Log Degrees")
        plt.ylabel("Log Probability")
        plt.show()
        
    def topAirportsFromIDs(self, idx_list):
        return [self.ID_to_Airport[self.ID_set[idx]] for idx in idx_list]


if __name__ == "__main__":
    cleaned_df = pd.read_csv("cleaned_flights.csv")
    flightNetwork = Network(cleaned_df)
    
    

    #plotting degree distribution
    start = 1
    end = 101
    direction = "out"
    flightNetwork.plot_power_fit(start,end,direction=direction)    
    
    #top in/out degree centralities 
    top_in = np.sum(flightNetwork.adjMatrix, axis=0)
    top_out = np.sum(flightNetwork.adjMatrix, axis=1)
    top_in_idx = np.argsort(top_in)[::-1]
    top_out_idx = np.argsort(top_out)[::-1]
    print(flightNetwork.topAirportsFromIDs(top_in_idx[:10]))
    print(flightNetwork.topAirportsFromIDs(top_out_idx[:10]))

    #eigenvalue centralities, does not work well
    adj = np.copy(flightNetwork.adjMatrix)
    adj[adj==0]=10e-10      #set 0s equal to a small value, may cause problems for airports with 0 outgoing flights
    norm = adj.sum(axis=1)
    adj = adj / norm    #make it be row stochastic
    [v,d]=np.linalg.eig(adj)
    top_eig = np.argsort(v)[::-1]
    top_airports = np.argsort(d[:,top_eig[0]])[::-1]
    k = 10
    print(flightNetwork.topAirportsFromIDs(top_airports[:k])) #show the top k airports
    
    #Katz
    [v,d]=np.linalg.eig(flightNetwork.adjMatrix)
    alpha = 0.5/max(abs(v))
    katz = np.linalg.inv(np.eye(len(adj))-adj*alpha).sum(axis = 1)
    print(flightNetwork.topAirportsFromIDs(np.argsort(katz)[::-1][:10]))
    
    #pagerank. Dunno if this works well. see my comment in google doc -Richard
    adj = np.copy(flightNetwork.adjMatrix)
    outdeg = adj.sum(axis = 1)
    outdeg[outdeg == 0] = 0.1
    D = np.diag(outdeg)
    ADinv=np.dot(adj,np.linalg.inv(D))
    [v,d]=np.linalg.eig(ADinv)
    alpha = 0.85/max(abs(v))
    pagerank = np.linalg.inv(np.eye(len(adj))-alpha*ADinv).sum(axis=1)
    pagerank2 = np.dot(D, np.linalg.inv(D-alpha*adj)).sum(axis = 1)
    print("PageRank:")
    print(flightNetwork.topAirportsFromIDs(np.argsort(pagerank)[::-1][:10]))
    
    #HITS
    AAT = np.dot(adj,np.transpose(adj))
    [v,d]=np.linalg.eig(AAT)
    top_eig = np.argsort(abs(v))[::-1]
    
    #authorities
    top_authority = np.argsort(d[:,top_eig[0]])[::-1]
    print(flightNetwork.topAirportsFromIDs(top_authority[:10]))
    
    #hubs
    top_hub = np.argsort(np.dot(np.transpose(adj),d[:,top_eig[0]]))[::-1]
    print(flightNetwork.topAirportsFromIDs(top_hub[:10]))
    
    
    #importance additional flight
    num_ports = len(adj)
    outdeg = adj.sum(axis = 1)
    outdeg[outdeg == 0] = 1
    connecting = np.zeros([num_ports, num_ports])
    for i in range(num_ports):
        for j in range(num_ports):
            #no flights between i and j
            total = 0
            if adj[i,j] == 0:
                #calculate how many people are expected to be traveling there
                scaled_out = adj[:,j]/outdeg
                total = np.dot(adj[i,:], scaled_out)
            connecting[i,j] = total
    
    top_connectors = []
    for i in range(num_ports):
        for j in range(num_ports):
            if i == j:
                connecting[i,j] = 0
    for i in range(num_ports):
        for j in range(num_ports):
            heapq.heappush(top_connectors, (connecting[i,j], i, j))
            if(len(top_connectors) > 40):
                heapq.heappop(top_connectors)

    top_connectors.sort()
    for connector in top_connectors[::-1]:
        (value, i, j) = connector
        print (flightNetwork.topAirportsFromIDs([i]),flightNetwork.topAirportsFromIDs([j]))

    ## DELAY ANALYSIS ##
    delay_summed = pd.read_csv("delay_summed.csv")
    delayNetwork = Network(delay_summed)
    
    #Find the top overall delayed flights
    top_delay_df = delayNetwork.df.sort_values('DELAY', ascending=False)
    print("Top delayed flights:")
    for i in range(10):
        print((top_delay_df['ORIGIN_AIRPORT_ID'].iloc[i],top_delay_df['DEST_AIRPORT_ID'].iloc[i],top_delay_df['DELAY'].iloc[i]))
    
    #Find the top delay-per-flights
    numFlightsToAnalyze = 50 #Look at top 50 delayed flights
    delayPerFlightDict = {}
    for i in range(numFlightsToAnalyze):
        origin = top_delay_df['ORIGIN_AIRPORT_ID'].iloc[i] 
        destination = top_delay_df['DEST_AIRPORT_ID'].iloc[i]
        delay = top_delay_df['DELAY'].iloc[i]
        org = flightNetwork.df[flightNetwork.df['ORIGIN'] == origin]
        
        delayPerFlightDict[(origin,destination)] = delay/sum(org[org['DEST'] == destination]['DEPARTURES_PERFORMED'])
    
    print("Top delay-per-flight flights:")
    sorted_delayPerFlight = sorted(delayPerFlightDict.items(), key=lambda x: x[1],reverse=True)
    print(sorted_delayPerFlight[:10])
        
    
    #Find the airports with the highest total delays
    delay_dict = {}
    for airportID in set(delayNetwork.df['ORIGIN_AIRPORT_ID']):
        a = delayNetwork.df[delayNetwork.df['ORIGIN_AIRPORT_ID']==airportID]
        delay_dict[airportID] = a['DELAY'].sum()

    sorted_delay_dict = sorted(delay_dict.items(), key=lambda x: x[1],reverse=True)
    print("Top delayed airports:")
    print(sorted_delay_dict[:10])
    
    #Find airports with the highest delay per flight
    delay_per_flight_dict = {}
    total_flight = {}
    top_delay_airports = sorted_delay_dict[:50] #We consider the top delayed airports.
    for aID in [top_delay_airports[i][0] for i in range(len(top_delay_airports))]:
        if aID in flightNetwork.df['ORIGIN'].values:
            a = flightNetwork.df[flightNetwork.df['ORIGIN']==aID]
            b = flightNetwork.df[flightNetwork.df['DEST']==aID]
            total_flight[aID] = a['DEPARTURES_PERFORMED'].sum() + b['DEPARTURES_PERFORMED'].sum()
            delay_per_flight_dict[aID] = delay_dict[aID]/(a['DEPARTURES_PERFORMED'].sum())

    sorted_delay_per_flight_dict = sorted(delay_per_flight_dict.items(), key=lambda x: x[1],reverse=True)
    print("Top delay-per-flight airports:")
    print(sorted_delay_per_flight_dict[:10])

