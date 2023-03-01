from numpy.ma.core import sqrt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy.core.numeric import ones

c1=0.25
c2=1.5
c3=0.5
c4=0.6
c5=5
beta=1
T=20
m=0.8
W_mi=0.15 #0.55
W_Q=0.65 # 0.85
delta_t=0.01
gamma = np.exp(- beta * delta_t)
sigma=0.24
sigma1=0.3

def get_next_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(Actions)
    else: #choose a random action
        return  Actions[np.argmin(Q_value.loc[state])]
        
def one_hot(States,state):
    if state not in States:
        return np.zeros(41)
    else:
        label_encoder = LabelEncoder()
        integer_encoded=label_encoder.fit_transform(States)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded[States.index(state)]
def step(State_0,action,delta_t):
    State_1=np.round(State_0+action*delta_t+sigma1*np.sqrt(delta_t)*np.random.normal(0,1),1)

    while State_1 not in States:
        State_1=np.round(State_0+action*delta_t+sigma1*np.sqrt(delta_t)*np.random.normal(0,1),1)

    return State_1

def get_reward(X,A,mi):
    m=np.dot(mi,States)
    return 0.5*A**2+c1*(X-c2*m)**2+c3*(X-c4)**2+c5*m**2
    
    
States=[round(x*0.1,2) for x in range(-15,26)]
temps=[round(x*0.01,2) for x in range(0,2001)]
Actions=[round(x*0.1,2) for x in range(-10,11)]
epsilon=0.15*ones(len(temps))


episodes=10000
mi_0=pd.DataFrame(1/len(States),States,temps)
Q_value=pd.DataFrame(0,States,Actions)
Nbre_times_state_action=pd.DataFrame(0,States,Actions)
Rho_mi=np.zeros((episodes,len(temps)))
Rho_Q=np.zeros((episodes,len(temps)))
QQ=np.zeros((len(States),len(Actions),episodes))
k=0
miT=np.zeros((len(States),episodes))
while (k < episodes):
    miT[:,k]=mi_0.loc[:,20]
    X_t0=np.random.choice(States,1,p=np.array(mi_0.loc[:,20]))[0]
    rho_Q=[]
    rho_mi=[]
    X=X_t0

    for n in range(len(temps)):
        old_Qvalue=Q_value
        tn=temps[n]
        rho_mi.append(1/((1+k)**W_mi))
        mi_0.loc[:,tn]=mi_0.loc[:,tn]+rho_mi[-1]*(one_hot(States,X)-mi_0.loc[:,tn])
        Action=get_next_action(X, epsilon[n])
        X_n1=step(X,Action,delta_t)
        Nbre_times_state_action.loc[X_n1,Action]+=1
        
        reward=get_reward(X,Action,mi_0.loc[:,tn])

        rho_Q.append(1/((1+Nbre_times_state_action.loc[X,Action])**W_Q))

        Q_value.loc[X,Action]=Q_value.loc[X,Action]+rho_Q[-1]*(reward+gamma*min(Q_value.loc[X_n1])-Q_value.loc[X,Action])
        X=X_n1
    
    #print('Qvalue:',np.sum(Q_value.mean()))
   # print('Old_Qvalue',np.sum(old_Qvalue.mean()))
    Rho_mi[k,:]=rho_mi
    Rho_Q[k,:]=rho_Q
    k=k+1

    print('k',k)
 

np.save('Nbre_times_state_action6.npy',Nbre_times_state_action.to_numpy())
np.save('Rho_mi6.npy',Rho_mi)
np.save('Rho_Q6.npy',Rho_Q)
np.save('Q_value6.npy',Q_value.to_numpy())
np.save('mi_06.npy',mi_0.to_numpy())
np.save('miT6.npy',miT)



