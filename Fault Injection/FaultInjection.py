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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Detection_Data():
    def __init__(self,L):
        # Setting signal length
        self.L = int(L)
        # Loading A2D2 data
        self.info = pd.read_csv('./A2D2 Raw Data/sen_info.csv')
        self.data_info = pd.read_csv('./A2D2 Raw Data/sen_info.csv').to_numpy()
        self.data_ap = pd.read_csv('./A2D2 Raw Data/data_ap.csv').to_numpy()
        self.data_sa = pd.read_csv('./A2D2 Raw Data/data_sa.csv').to_numpy()
        self.data_bp = pd.read_csv('./A2D2 Raw Data/data_bp.csv').to_numpy()
        
        t = [min(self.data_info[0,1:]), min(self.data_info[1,1:]), min(self.data_info[2,1:])]
        # Number of signals for each city
        self.n = np.round(np.divide(t,L)).astype(int)
        
        # Taking the FSO attribute = (Max - Min)/2
        ap_fso = (max(self.data_ap[:,1:].flatten()) - min(self.data_ap[:,1:].flatten()))/2
        sa_fso = (max(self.data_sa[:,1:].flatten()) - min(self.data_sa[:,1:].flatten()))/2
        bp_fso = (max(self.data_bp[:,1:].flatten()) - min(self.data_bp[:,1:].flatten()))/2
        self.fso = np.array([ap_fso, sa_fso, bp_fso]) 

        # Decomposing Cities and adding labels and columns
        ap1 = pd.DataFrame(self.data_ap[0:self.n[0]*L,1])
        ap2 = pd.DataFrame(self.data_ap[0:self.n[1]*L,2])
        ap3 = pd.DataFrame(self.data_ap[0:self.n[2]*L,3])

        sa1 = pd.DataFrame(self.data_sa[0:self.n[0]*L,1])
        sa2 = pd.DataFrame(self.data_sa[0:self.n[1]*L,2])
        sa3 = pd.DataFrame(self.data_sa[0:self.n[2]*L,3])

        bp1 = pd.DataFrame(self.data_bp[0:self.n[0]*L,1])
        bp2 = pd.DataFrame(self.data_bp[0:self.n[1]*L,2])
        bp3 = pd.DataFrame(self.data_bp[0:self.n[2]*L,3])

        index1 = pd.DataFrame([x for x in range(self.n[0]*L)])
        index2 = pd.DataFrame([x for x in range(self.n[1]*L)])
        index3 = pd.DataFrame([x for x in range(self.n[2]*L)])

        city1 = pd.DataFrame([1 for x in range(self.n[0]*L)])
        city2 = pd.DataFrame([2 for x in range(self.n[1]*L)])
        city3 = pd.DataFrame([3 for x in range(self.n[2]*L)]) 

        zeros1 = pd.DataFrame([0 for x in range(self.n[0]*L)])
        zeros2 = pd.DataFrame([0 for x in range(self.n[1]*L)])
        zeros3 = pd.DataFrame([0 for x in range(self.n[2]*L)])

        case1 = pd.DataFrame([0 for x in range(self.n[0]*L)])
        case2 = pd.DataFrame([0 for x in range(self.n[1]*L)])
        case3 = pd.DataFrame([0 for x in range(self.n[2]*L)])

        signal_index = []
        for i in range(self.n[0]):
            index = pd.DataFrame([i for x in range(L)])
            signal_index.append(index)
        signal_index1 = pd.concat(signal_index, ignore_index=True)

        signal_index = []
        for i in range(self.n[1]):
            index = pd.DataFrame([i for x in range(L)])
            signal_index.append(index)
        signal_index2 = pd.concat(signal_index, ignore_index=True)

        signal_index = []
        for i in range(self.n[2]):
            index = pd.DataFrame([i for x in range(L)])
            signal_index.append(index)
        signal_index3 = pd.concat(signal_index, ignore_index=True)

        titles = ['idx', 'ap_val', 'sa_val', 'bp_val', 'city_id', 'signal_idx', 'case', 'fault_cat', 'healthy/faulty', 'ap_fault', 'sa_fault', 'bp_fault']

        frames1 = [index1, ap1, sa1, bp1, city1, signal_index1, case1, zeros1, zeros1, zeros1, zeros1, zeros1]
        frames2 = [index2, ap2, sa2, bp2, city2, signal_index2, case2, zeros2, zeros2, zeros2, zeros2, zeros2]
        frames3 = [index3, ap3, sa3, bp3, city3, signal_index3, case3, zeros3, zeros3, zeros3, zeros3, zeros3]

        self.h_city1 = pd.concat(frames1, axis=1)
        self.h_city2 = pd.concat(frames2, axis=1)
        self.h_city3 = pd.concat(frames3, axis=1)

        self.h_city1.columns= titles
        self.h_city2.columns= titles
        self.h_city3.columns= titles

    def data2signal(self,data,city):
        ap = np.reshape(data[:self.n[city]*self.L,0],(self.n[city], self.L))
        sa = np.reshape(data[:self.n[city]*self.L,1],(self.n[city], self.L))
        bp = np.reshape(data[:self.n[city]*self.L,2],(self.n[city], self.L))
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
  
    def inject_erratic(self, data, sensor='ap'):
        if sensor == 'ap':
            ch = 'ap_val'
            ch_f = 'ap_fault'
            fso = self.fso[0]
        elif sensor == 'sa':
            ch = 'sa_val'
            ch_f = 'sa_fault'
            fso = self.fso[1]
        else:
            ch = 'bp_val'
            ch_f = 'bp_fault'
            fso = self.fso[2]

        faulty_signal = data.copy()
        seq = data[ch]
        L = len(seq)
        idx1 = np.random.randint(low=1, high=L*0.6)
        idx2 = np.random.randint(low=idx1, high=L)
        for j in range(idx1,idx2):
            r = np.random.normal(0,0.2)
            erratic = np.round(r*fso,1)
            faulty_signal[ch].iloc[j] += erratic
            faulty_signal['healthy/faulty'].iloc[j] = 1
            faulty_signal[ch_f].iloc[j] = 1

        return faulty_signal
    
    def inject_drift(self, data, sensor='ap'):
        if sensor == 'ap':
            ch = 'ap_val'
            ch_f = 'ap_fault'
            fso = self.fso[0]
        elif sensor == 'sa':
            ch = 'sa_val'
            ch_f = 'sa_fault'
            fso = self.fso[1]
        else:
            ch = 'bp_val'
            ch_f = 'bp_fault'
            fso = self.fso[2]

        faulty_signal = data.copy()
        seq = data[ch]
        L = len(seq)
        idx1 = 0    
        #idx1 = np.random.randint(low=1, high=L*0.2)
        #idx2 = np.random.randint(low=L*0.6, high=L)
        idx2 = L
        delta = idx2 - idx1
        r1 = 0.01 #np.random.normal(0.1,0.09)
        r2 = 1 #np.random.normal(1.2,0.9)
        drift_grad = r1*(fso/delta)
        for j in range(delta):
            drift = faulty_signal[ch].iloc[idx1+j]+ drift_grad*j+r2*fso
            if drift > 2*fso:
                faulty_signal[ch].iloc[idx1+j] = 2*fso
            else:
                faulty_signal[ch].iloc[idx1+j] = drift

            faulty_signal['healthy/faulty'].iloc[idx1+j] = 1
            faulty_signal[ch_f].iloc[idx1+j] = 2

        return faulty_signal
    
    def inject_hardover(self, data, sensor='ap'):
        if sensor == 'ap':
            ch = 'ap_val'
            ch_f = 'ap_fault'
            fso = self.fso[0]
        elif sensor == 'sa':
            ch = 'sa_val'
            ch_f = 'sa_fault'
            fso = self.fso[1]
        else:
            ch = 'bp_val'
            ch_f = 'bp_fault'
            fso = self.fso[2]

        faulty_signal = data.copy()
        seq = data[ch]
        L = len(seq) 

        idx1 = np.random.randint(low=1, high=L/4)+int(L/4)
        delta = np.random.randint(low=1, high=L/4)+int(L/4)
        #sign = np.random.choice((-1,1))
        for j in range(delta):
            faulty_signal[ch].iloc[idx1+j] = fso*2*1.1
            faulty_signal['healthy/faulty'].iloc[idx1+j] = 1
            faulty_signal[ch_f].iloc[idx1+j] = 3

        return faulty_signal
    
    def inject_spike(self, data, sensor='ap'):
        if sensor == 'ap':
            ch = 'ap_val'
            ch_f = 'ap_fault'
            fso = self.fso[0]
        elif sensor == 'sa':
            ch = 'sa_val'
            ch_f = 'sa_fault'
            fso = self.fso[1]
        else:
            ch = 'bp_val'
            ch_f = 'bp_fault'
            fso = self.fso[2]

        faulty_signal = data.copy()
        seq = data[ch]
        L = len(seq) 
        num = np.random.randint(low=5, high=10)
        for j in range(num):
            loc = np.random.randint(low=1, high=L-1)
            #sign = np.random.choice((-1,1))
            faulty_signal[ch].iloc[loc] += fso
            faulty_signal['healthy/faulty'].iloc[loc] = 1
            faulty_signal[ch_f].iloc[loc] = 4

        return faulty_signal
    
    def viz_fault(self, h_seq, f_seq, fso, draw_fso=False):
        t = np.linspace(0,len(h_seq)/self.L, len(h_seq))
        plt.plot(t, f_seq, 'r')
        plt.plot(t, h_seq, 'b')
        if draw_fso:
            plt.hlines([0.2*fso, -0.2*fso], 0, len(h_seq)/self.L, linestyles='dashed')
        plt.show()
        
    def make_fault(self, data, inject_error= inject_erratic, case=8):

        # Initialize fault Cases
        f11_dataset = data.copy()
        f12_dataset = data.copy()
        f13_dataset = data.copy()
        f21_dataset = data.copy()
        f22_dataset = data.copy()
        f23_dataset = data.copy()
        f33_dataset = data.copy()
        
        f11_dataset['case'] = pd.DataFrame([1 for x in range(len(data))])
        f12_dataset['case'] = pd.DataFrame([2 for x in range(len(data))])
        f13_dataset['case'] = pd.DataFrame([3 for x in range(len(data))])
        f21_dataset['case'] = pd.DataFrame([4 for x in range(len(data))])
        f22_dataset['case'] = pd.DataFrame([5 for x in range(len(data))])
        f23_dataset['case'] = pd.DataFrame([6 for x in range(len(data))])
        f33_dataset['case'] = pd.DataFrame([7 for x in range(len(data))])

        # Healthy Case
        self.h_dataset = data.copy()

        # Injecting error to each signal
        index = data['signal_idx']
        for i in range(index.iloc[-1]+1):
            # Case 1: FHH
            f11_dataset.loc[index==i] = inject_error(f11_dataset.loc[index==i], sensor='ap')

            # Case 2: HFH
            f12_dataset.loc[index==i] = inject_error(f12_dataset.loc[index==i], sensor='sa')

            # Case 3: HHF
            f13_dataset.loc[index==i] = inject_error(f13_dataset.loc[index==i], sensor='bp')

            # Case 4: FFH
            f21_dataset.loc[index==i] = inject_error(f21_dataset.loc[index==i], sensor='ap')
            f21_dataset.loc[index==i] = inject_error(f21_dataset.loc[index==i], sensor='sa')

            # Case 5: HFF
            f22_dataset.loc[index==i] = inject_error(f22_dataset.loc[index==i], sensor='sa')
            f22_dataset.loc[index==i] = inject_error(f22_dataset.loc[index==i], sensor='bp')

            # Case 6: FHF
            f23_dataset.loc[index==i] = inject_error(f23_dataset.loc[index==i], sensor='ap')
            f23_dataset.loc[index==i] = inject_error(f23_dataset.loc[index==i], sensor='bp')

            # Case 7: FFF
            f33_dataset.loc[index==i] = inject_error(f33_dataset.loc[index==i], sensor='ap')
            f33_dataset.loc[index==i] = inject_error(f33_dataset.loc[index==i], sensor='sa')
            f33_dataset.loc[index==i] = inject_error(f33_dataset.loc[index==i], sensor='bp')

        # Pulling all the faulty combinations together

        sets = [f11_dataset, f12_dataset, f13_dataset, f21_dataset, f22_dataset, f23_dataset, f33_dataset]
        f_dataset = pd.concat(sets, ignore_index=True)   

        # Case selection
        dataset = {
            0: lambda x: self.h_dataset,
            1: lambda x: f11_dataset,
            2: lambda x: f12_dataset,
            3: lambda x: f13_dataset,
            4: lambda x: f21_dataset,
            5: lambda x: f22_dataset,
            6: lambda x: f23_dataset,
            7: lambda x: f33_dataset,
            8: lambda x: f_dataset,
        }[case](1)
        
        return dataset

    def make_city_dataset(self, city):
        # Set the city
        if city==1:
            data = self.h_city1
        elif city==2:
            data = self.h_city2
        else:
            data = self.h_city3

        # Adding the 4 fault types to the city
        self.erratic_dataset = self.make_fault(data, inject_error= self.inject_erratic)
        self.erratic_dataset['fault_cat'] =  pd.DataFrame([1 for x in range(len(self.erratic_dataset))])

        self.drift_dataset = self.make_fault(data, inject_error= self.inject_drift)
        self.drift_dataset['fault_cat'] = pd.DataFrame([2 for x in range(len(self.drift_dataset))])

        self.hardover_dataset = self.make_fault(data, inject_error= self.inject_hardover)
        self.hardover_dataset['fault_cat'] = pd.DataFrame([3 for x in range(len(self.drift_dataset))])

        self.spike_dataset = self.make_fault(data, inject_error= self.inject_spike)
        self.spike_dataset['fault_cat'] = pd.DataFrame([4 for x in range(len(self.drift_dataset))])

        # Stacking them together
        sets = [self.erratic_dataset, self.drift_dataset, self.hardover_dataset, self.spike_dataset]
        datasets = pd.concat(sets, ignore_index=True)

        return datasets


    def csv_out(self, dataset, path):
        n = dataset.shape[0]
        c = dataset.shape[1]
        L = dataset.shape[2]
        dataset_out = pd.DataFrame(np.zeros((n*L,c)))
        for i in range(c):
            dataset_out.iloc[:,i] = np.reshape(dataset[:,i,:],(n*L))
        dataset_out.to_csv(path)
    
    