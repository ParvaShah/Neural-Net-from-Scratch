# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:27:07 2018

@author: PARVA SHAH
"""


import math as m


def create_layer_and_neurons(n):
    
    
	num_layers=len(n)
    
   
   
    total_weights=0
    for i in range(len(n)-1):
        total_weights+=(n[i])*(n[i+1])  
        #print('tw',total_weights)
       
        nedict={i:n[i] for i in range(num_layers)}  
        print(nedict)
        
    from random import random
    #seed(1)
    x=[]
    for i in range(total_weights):
        x.append(round(random(),4))
        #x.append(1)
   
      
    bval=[]
    for i in range(num_layers-1):
        bval.append(1)
    
    bwgt=[]
    for i in range(num_layers-1):
        bwgt.append(round(random(),4))
    
    start=0
    t=0
   
    wedict={}
  
    newe=[]
    # Assign weights to each neuron which travels to next neuron 
    for l in range(num_layers-1):
        for ne in range(n[l]):
            newe=x[start:n[l+1]+start]
       
            wedict.update({t:newe})
           
            t+=1
            start+=n[l+1]
    
    
    return nedict,wedict,n,bval,bwgt


def forward_pass(network,weights,inputs,n,bval,bwgt):
    
    output={}
  
    delt=0
           
    neurused=0
    op=-1
    for i in range(len(network)):
        for x in range(network[i]) :
            add=0.0
            if i == 0:
                 output.update({x:inputs[x]})
                 #print(output)
                 op+=1
            
            else:
                no_neur=network[i-1]
        
               
                
                for neu in range(no_neur):
                    #print('in create',neu,no_neur,x)
                    #print(weights[neurused-no_neur+neu][x],'',output[neurused-no_neur+neu])
                    add+=weights[neurused-no_neur+neu][x]*output[neurused-no_neur+neu]
                op+=1
                
                add+=bval[i-1]*bwgt[i-1] 
                delt=sigmoid(add)
                output.update({op:delt})
                #print('output',output)
        neurused+=network[i]
   
       
    
    
   
    return output             

def sigmoid(add):
    #Sigmoid
    return 1/(1+m.exp(-1*(add)))
    
#Relu
def relu(add):    
    if add>0:
        return add
    else:
        return 0
    

def backprop_delta(network,weights,n,bval,bwgt,output,label):
    #print(network[len(network)-1])
    length=len(network)
    real=label
   # print("real",real)
    itert=0
    error={}
    neitr=len(weights)+network[len(network)-1]-1########Tricky
    for i in range (length-1,-1,-1):
        #x=i
        k=0
        for j in range (network[i]):
            
            if i==length-1:
               # print('d','i','y','neitr','itert',i,neitr,itert,x)
                #print('one',output[neitr]-(1-output[neitr]))
                #print('tone',real-output[neitr])
                #print("you know",(float)(real[k])-output[neitr],output[neitr],real[k])
                calerr=(output[neitr]*(1-output[neitr]))*(real[k]-output[neitr])
               # print('cal',calerr)
                error.update({itert:calerr})
               # print(error)
               # print('kkkwk')
                itert+=1
                #print(error)
                neitr-=1
                k+=1
               # print(k)
            
            else:
                neer=0
                d=len(weights[neitr])-1
                for k in range (len(weights[neitr])):
                    #y=(len(weights[neitr])-1+(itert-1))
                    y= len(error)-(len(weights[neitr]))
                   # print('kkkwk121',d)
                    #print('d',d,'i',i,'y',y,'neitr',neitr,'itert',itert,'j',j,'x',x)
                    neer+=weights[neitr][d]*error[y]
                    d-=1
                    y+=1
                calerr=(output[neitr]*(1-output[neitr]))*(neer)
                #print("you know")
                error.update({itert:calerr})
                
                
                
                itert+=1
               # print('qww',error)
                neitr-=1
   # print('error',error)
    reverr(error)
    #updateWeight(error,weights)
    weights,bwgt=backprop_updateWeight(error,weights,network,bwgt)
    return weights,bwgt
   
def reverr(x):
    temp=0
    i=0
    j=len(x)-1
   
    while i<=j:
        
        temp=x[i]
        x[i]=x[j]
        x[j]=temp
        i+=1
        j-=1
        #print(x)
    

     
     
def backprop_updateWeight(error,weights,network,bwgt):
    incner=0
    incwgt=0
    learningRate=0.1
    for i in range (len(network)-1): #because we have to iterarte through just 
                        #neurons which have weights. Last neuron doent have weight
        d=incner+(network[i])
        for j in range(network[i]):
            
            incwgt=0
            
            for l in range(d,d+network[i+1]):
                weights[incner][incwgt]+= learningRate*output[incner]*error[l]
                #print('incner,incwgt,l,d',incner,incwgt,l,d)
                incwgt+=1
            incner+=1
    bwgt=updateBias(network,error,learningRate,bwgt)
    return weights,bwgt     
    
def updateBias(network,error,learningRate,bwgt):
    
    op=0
    bval=0
    x=[]
    for i in range(len(network)):
        for j in range(network[i]):
            if i == 0:
                op+=1
                continue
            else:
                x.append(error[op])
                op+=1
            avg= sum(x) / float(len(x))
            bwgt[bval]+=learningRate*avg
    return bwgt


from copy import deepcopy


    
            
initialweights={}    
network={}
weights={}
n=[]
bval={}
bwgt={}
inputs=[1,2]
network,weights,n,bval,bwgt=create_layer_and_neurons([4,10,6,3])
x1=deepcopy(weights)
bv=deepcopy(bwgt)
#output=forward_pass(network,weights,inputs,n,bval,bwgt)
#initialweights=deepcopy(weights)

#backprop_delta(network,weights,inputs,n,bval,bwgt,output)
#data=[[2.7810836,2.550537003],[1.465489372,2.362125076],[3.396561688,4.400293529],[1.38807019,1.850220317],[3.06407232,3.005305973],[7.627531214,2.759262235]]
#label=[0,0,0,0,1,1]	


##########
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :]  # we only take the first two features.
x = iris.target
#print(X)

#########
############3


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
#data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
#values = array(data)
#print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(x)
#print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#print(onehot_encoded)
label=onehot_encoded
#print(label)

#############


print(len(X))


one=[]
p=[]
q=[]
flag=1
for epoc in range(2):
    print('Iteration:',epoc)
    for i in range(len(X)):
        output=forward_pass(network,weights,X[i],n,bval,bwgt)    
    #print(output)
    #print('in range')
    #print(output[len(output)-1])
        #one.append(output[len(output)-1])
        one=output[len(output)-1]
        #print('one',one)
        two=output[len(output)-2]
        three=output[len(output)-3]
        print(one,two,three)
        p.append(one)
        q.append(two)
        d=max((one,two,three))
        '''if epoc>10 and p[i]==p[i-1]:
            flag=0
            break'''
        '''
    #print('two',two)
        one.append(two)
        result=max(one)
        '''
        if d==one:
            d=1
        elif d==two:
            d=2
        else:
            d=3
            
        print('result-',d,' label-',label[i])
        #print('result-',d,' label-',label[i])
        
        initialweights=deepcopy(weights)
        weights,bwgt=backprop_delta(network,weights,n,bval,bwgt,output,label[i])
        #if epoc==1:
         #   just(weights)
    if flag==0:
        break
print("outside")
#print(x)    
#print(weights)


