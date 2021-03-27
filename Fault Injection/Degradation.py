'''
Creating a class that loads the healthy datapoints and apply different methods on them.
Signals are assumed to have no overlaps and L is the desired signal length.
methods:
take_h_dataset : takes the healthy datapoints in and creates healthy signals for all the 3 sensors
data2signal : takes in datapoints for a specified city and turn them into signals for all the 3 sensors
signal2data: takes in signals and reshape them into consecutive datapoints
inject_lin_erratic : takes in a sequence of datapoints and adds a linear degradation of erratic fault type
inject_erratic : takes in a signal and injects fault into it
inject_drift:
inject_hardover:
inject_spike:

viz_fault: takes in a healthy signal and faulty signal and plot them together
make_fault: takes in a set of healthy signals and adds a specified type of fault to the sensors in all the combinations possible or even specific cases of combinations
degrade : takes in the datapoints of a desired city and degrades the data sequences with all the combinations
csv_out: takes in the dataset signals and convert them into datapoints and export to csv


'''
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Prediction_Data():
    def __init__(self,L):
        # Setting signal length
        self.L = int(L)

        # Loading A2D2 data (extended)
        self.data_ap = pd.read_csv('./A2D2 Raw Data/data_ap_ext.csv').to_numpy()
        self.data_sa = pd.read_csv('./A2D2 Raw Data/data_sa_ext.csv').to_numpy()
        self.data_bp = pd.read_csv('./A2D2 Raw Data/data_bp_ext.csv').to_numpy() 
        t = len(self.data_ap)

        # Number of signals for each city
        self.n = np.round(t/L).astype(int)

        # Taking the FSO attribute = (Max - Min)/2
        ap_fso = (max(self.data_ap[:,1:].flatten()) - min(self.data_ap[:,1:].flatten()))/2
        sa_fso = (max(self.data_sa[:,1:].flatten()) - min(self.data_sa[:,1:].flatten()))/2
        bp_fso = (max(self.data_bp[:,1:].flatten()) - min(self.data_bp[:,1:].flatten()))/2
        self.fso = np.array([ap_fso, sa_fso, bp_fso])

    def take_h_dataset(self):
        ap1 = np.reshape(self.data_ap[0:self.n*self.L,1],(self.n,self.L))
        ap2 = np.reshape(self.data_ap[0:self.n*self.L,2],(self.n,self.L))
        ap3 = np.reshape(self.data_ap[0:self.n*self.L,3],(self.n,self.L))
        dataset_ap = np.vstack((ap1,ap2,ap3))

        sa1 = np.reshape(self.data_sa[0:self.n*self.L,1],(self.n,self.L))
        sa2 = np.reshape(self.data_sa[0:self.n*self.L,2],(self.n,self.L))
        sa3 = np.reshape(self.data_sa[0:self.n*self.L,3],(self.n,self.L))
        dataset_sa = np.vstack((sa1,sa2,sa3))

        bp1 = np.reshape(self.data_bp[0:self.n*self.L,1],(self.n,self.L))
        bp2 = np.reshape(self.data_bp[0:self.n*self.L,2],(self.n,self.L))
        bp3 = np.reshape(self.data_bp[0:self.n*self.L,3],(self.n,self.L))
        dataset_bp = np.vstack((bp1,bp2,bp3))

        self.h_dataset = np.stack((dataset_ap, dataset_sa, dataset_bp), axis=1)
        return self.h_dataset

    def data2signal(self,data):
        ap = np.reshape(data[:self.n*self.L,0],(self.n, self.L))
        sa = np.reshape(data[:self.n*self.L,1],(self.n, self.L))
        bp = np.reshape(data[:self.n*self.L,2],(self.n, self.L))
        signals = np.stack((ap,sa,bp), axis=1)
        return signals

    def signal2data(self, dataset):
        n = dataset.shape[0]
        c = dataset.shape[1]
        L = dataset.shape[2]
        data = np.zeros((n*L,c))
        for i in range(c):
            data[:,i] = np.reshape(dataset[:,i,:],(n*L))
        return data

    def inject_lin_erratic(self, data, fso):
        faulty_data = np.array(data)
        delta = len(data)
        idx1 = int(0.1*delta)
        idx2 = delta - idx1
        for j in range(idx2):
            r = np.random.normal(0,0.01)
            erratic = r*fso*35*(j/idx2)
            faulty_data[idx1+j] += np.round(erratic,1)
        return faulty_data

    def inject_exp_erratic(self, data, fso):
        faulty_data = np.array(data)
        delta = len(data)
        idx1 = int(0.1*delta)
        idx2 = delta - idx1
        for j in range(idx2):
            r = np.random.normal(0,0.01)
            erratic = r*fso*(1/750)*math.exp(5*((j/idx2)+1.06))
            faulty_data[idx1+j] += np.round(erratic,1)
        return faulty_data

    def inject_sin_erratic(self, data, fso):
        faulty_data = np.array(data)
        L = len(data)
        idx1 = int(0.1*L)
        idx2 = L - idx1
        for j in range(idx2):
            r = np.random.normal(0,0.01)
            erratic = r*fso*35*math.sin(4*math.pi*(j/idx2))
            faulty_data[idx1+j] += np.round(erratic,1)
        return faulty_data

    def inject_lin_drift(self, data, fso):
        faulty_data = np.array(data)
        L = len(data)
        idx1 = int(0.1*L)
        idx2 = L - idx1
        for j in range(idx2):
            grad = 7*fso #*(j/idx2)
            drift = grad*(j/idx2)
            faulty_data[idx1+j] += np.round(drift,1)
        return faulty_data

    def inject_exp_drift(self, data, fso):
        faulty_data = np.array(data)
        L = len(data)
        idx1 = int(0.1*L)
        idx2 = L - idx1
        for j in range(idx2):
            grad = (1/1000)*fso*math.exp(5*((j/idx2)+0.8))
            #grad = 7*fso #*(j/idx2)
            drift = grad*(j/idx2)
            faulty_data[idx1+j] += np.round(drift,1)
        return faulty_data

    def inject_lin_drift_old(self, data, fso):
        faulty_data = np.array(data)
        L = len(data)
        SL = 240
        a = 240
        b = int(0.1*L)
        xmax = int((L-SL-b)/a)
        idx1 = [int(a*x+b) for x in range(xmax+1)]
        for loc in idx1:
            delta = np.random.randint(low=0.4*SL, high=0.5*SL)
            grad = 1.3*fso*(loc/idx1[-1])
            for j in range(delta):
                creep = grad*(j/delta)
                faulty_data[loc+j] += np.round(creep,1)

        return faulty_data

    def inject_exp_drift_old(self, data, fso):
        faulty_data = np.array(data)
        L = len(data)
        SL = 240
        a = 240
        b = int(0.1*L)
        xmax = int((L-SL-b)/a)
        idx1 = [int(a*x+b) for x in range(xmax+1)]
        for loc in idx1:
            delta = np.random.randint(low=0.4*SL, high=0.5*SL)
            grad = (1/1000)*fso*math.exp(5*((loc/idx1[-1])+0.4487))
            for j in range(delta):
                creep = grad*(j/delta)
                faulty_data[loc+j] += np.round(creep,1)

        return faulty_data

    def inject_lin_spike(self, seq, fso):
        faulty_seq = np.array(seq)
        L = len(seq)
        SL = 240
        a = 240
        b = int(0.1*L)
        xmax = int((L-b)/a)
        idx1 = [int(a*x+b) for x in range(xmax)]
        for loc in idx1:
            num = np.random.randint(low=4,high=9)
            loc_pikes = [np.random.randint(low=0.15*SL , high= SL-1) for x in range(num)]
            faulty_seq[loc] += 1.2*fso*(loc/idx1[-1])
            for i in loc_pikes:
                faulty_seq[loc+i] += 1.2*fso*(loc/idx1[-1])

        return faulty_seq

    def inject_sin_spike(self, seq, fso):
        faulty_seq = np.array(seq)
        L = len(seq)
        SL = 240
        a = 240
        b = int(0.1*L)
        xmax = int((L-b)/a)
        idx1 = [int(a*x+b) for x in range(xmax)]
        for loc in idx1:
            num = np.random.randint(low=4,high=9)
            loc_pikes = [np.random.randint(low=0.15*SL , high= SL-1) for x in range(num)]
            faulty_seq[loc] += abs(1*fso*math.cos(4*math.pi*(loc/idx1[-1])))
            for i in loc_pikes:
                faulty_seq[loc+i] += abs(1*fso*math.cos(4*math.pi*(loc/idx1[-1])))

        return faulty_seq

    def inject_exp_spike(self, seq, fso):
        faulty_seq = np.array(seq)
        L = len(seq)
        SL = 240
        a = 240
        b = int(0.1*L)
        xmax = int((L-b)/a)
        idx1 = [int(a*x+b) for x in range(xmax)]
        for loc in idx1:
            num = np.random.randint(low=4,high=9)
            loc_pikes = [np.random.randint(low=0.15*SL , high= SL-1) for x in range(num)]
            faulty_seq[loc] += (1/1000)*fso*math.exp(5*((loc/idx1[-1])+0.4487))
            for i in loc_pikes:
                faulty_seq[loc+i] += (1/1000)*fso*math.exp(5*((loc/idx1[-1])+0.3687))

        return faulty_seq


    def viz_fault(self, h_data, f_data, fso, draw_fso=True):
        t = np.linspace(0,len(h_data)/self.L, len(h_data))
        plt.plot(t, f_data, 'r')
        plt.plot(t, h_data, 'b')
        if draw_fso:
            plt.hlines([0.2*fso, -0.2*fso], 0, len(h_data)/self.L, linestyles='dashed')
        plt.show()

    def make_fault(self, data, inject_error= inject_exp_erratic, case=8):
        # Initialize fault Cases
        self.f11_data = np.array(data)
        self.f12_data = np.array(data)
        self.f13_data = np.array(data)
        self.f21_data = np.array(data)
        self.f22_data = np.array(data)
        self.f23_data = np.array(data)
        self.f33_data = np.array(data)
        
        # Healthy Case
        self.h_data = data
        # Injecting error to each signal
        # Case 1: FHH
        self.f11_data[:,0] = inject_error(self, self.f11_data[:,0],self.fso[0])
        # Case 2: HFH
        self.f12_data[:,1] = inject_error(self, self.f12_data[:,1],self.fso[1])
        # Case 3: HHF
        self.f13_data[:,2] = inject_error(self, self.f13_data[:,2],self.fso[2])
        # Case 4: FFH
        self.f21_data[:,0] = inject_error(self, self.f21_data[:,0],self.fso[0])
        self.f21_data[:,1] = inject_error(self, self.f21_data[:,1],self.fso[1])
        # Case 5: HFF
        self.f22_data[:,1] = inject_error(self, self.f22_data[:,1],self.fso[1])
        self.f22_data[:,2] = inject_error(self, self.f22_data[:,2],self.fso[2])
        # Case 6: FHF
        self.f23_data[:,0] = inject_error(self, self.f23_data[:,0],self.fso[0])
        self.f23_data[:,2] = inject_error(self, self.f23_data[:,2],self.fso[2])
        # Case 7: FFF
        self.f33_data[:,0] = inject_error(self, self.f33_data[:,0], self.fso[0])
        self.f33_data[:,1] = inject_error(self, self.f33_data[:,1], self.fso[1])
        self.f33_data[:,2] = inject_error(self, self.f33_data[:,2], self.fso[2])

        # Pulling all the faulty combinations together    
        self.f_data = np.vstack((self.f11_data, self.f12_data, self.f13_data,
                                    self.f21_data, self.f22_data, self.f23_data,
                                    self.f33_data))

        # Case selection
        self.dataset = {
            0: lambda x: self.h_data,
            1: lambda x: self.f11_data,
            2: lambda x: self.f12_data,
            3: lambda x: self.f13_data,
            4: lambda x: self.f21_data,
            5: lambda x: self.f22_data,
            6: lambda x: self.f23_data,
            7: lambda x: self.f33_data,
            8: lambda x: self.f_data,
        }[case](1)
        
        return self.dataset

    def degrade(self, city= 0, case=0, degradation_func= inject_lin_spike):
        data = np.stack((self.data_ap[:self.n*self.L, city+1],
                         self.data_sa[:self.n*self.L, city+1],
                         self.data_bp[:self.n*self.L, city+1]),
                         axis=1)
        datapoints = self.make_fault(data, inject_error= degradation_func, case= case)
        dataset = self.data2signal(datapoints)
        
        return dataset