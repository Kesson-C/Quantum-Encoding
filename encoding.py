# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 21:35:22 2021

@author: kesson
"""
import numpy as np
from bitarray import bitarray
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator
from copy import copy 
import time

def label2Pauli(s): # can be imported from groupedFermionicOperator.py
    """
    Convert a Pauli string into Pauli object. 
    Note that the qubits are labelled in descending order: 'IXYZ' represents I_3 X_2 Y_1 Z_0
    
    Args: 
        s (str) : string representation of a Pauli term
    
    Returns:
        qiskit.quantum_info.Pauli: Pauli object of s
    """
    
    xs = []
    zs = []
    label2XZ = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
    for c in s[::-1]:
        x, z = label2XZ[c]
        xs.append(x)
        zs.append(z)
    return Pauli(z = zs, x = xs)

def complete_mapping(naive_mapping):
    mapping = copy(naive_mapping)
    for k, w in naive_mapping.items():
        k1, k2 = k
        if (k1 != k2) and ((k2, k1) not in naive_mapping.keys()):
            new_k = (k2, k1)
            paulis = list(map(lambda x: [np.conj(x[0]), x[1]], w.paulis))
            mapping[new_k] = WeightedPauliOperator(paulis)
    return mapping

def bitmasks(n,m):
    """
    (a math tool)
    Return a list of numbers with `m` ones in their binary expressions in ascending order. 
    The last element in the list does not exceed `until`.
    
    Args:
        n (int) : number of all numbers
        m (int) : number of ones
    
    Returns:
        ([Int]) : the generated list
    """
    if m < n:
        if m > 0:
            for x in bitmasks(n-1,m-1):
                yield bitarray([1]) + x
            for x in bitmasks(n-1,m):
                yield bitarray([0]) + x
        else:
            yield n * bitarray('0')
    else:
        yield n * bitarray('1')
        
def to10(C,n):
    C10=[]
    for i in C:
        tn=n
        tv=0
        for j in i:
            tn-=1
            if j=='1':
                tv+=2**tn
        C10.append(tv)
    return np.array(C10)

def to2(MAPC,n):
    C2=[]
    for i in MAPC:
        C="{0:b}".format(i)
        C=C.zfill(n)
        C2.append(C)
    return C2

def gop(n):
    """
    (a math tool)
    Return a list of numbers with `n` ones in their binary expressions in ascending order. 
    The last element in the list does not exceed `until`.
    
    Args:
        n (int) : number of ones
        until (int) : upper bound of list elements
    
    Returns:
        ([Int]) : the generated list
    """
    OP=[]
    for i in range(n):
        for j in range(i+1,n):
            OP.append(2**i+2**j)
    return np.array(OP)

def coset(CV,GOP):
    ALL=[]
    for i in GOP:
        SUB=[]
        Ctmp=CV^i
        for j in range(len(CV)):
            if CV[j] in Ctmp:
                k=np.where(Ctmp==CV[j])[0][0]
                if k>j:
                    SUB.append([j,k])
        ALL.append(SUB)
    return ALL

def Bextend(Basis,bn):
    B=np.array(Basis)
    for i in range(bn-1):
        C=B[i]^B[(i+1):]
        Basis.extend(list(C))        

def cosetop(bn,sn):
    Basis=[]
    for i in range(bn):
        Basis.append(2**i)
        
    B=np.array(Basis)
    Bextend(Basis,len(Basis))
    if len(B)==sn-1:
        return np.array(Basis),bn
    
    breaker=1
    while(breaker):
        for i in range(2**bn):
            sum=0
            for j in B^i:
                sum+=(j in Basis)
            if sum==0:
                B=np.insert(B,len(B),i)
                break
            if i==2**bn-1:
                breaker=0
        Basis=list(B)
        Bextend(Basis,len(Basis))
      
        if len(B)==sn-1:
            return np.array(Basis),bn
    return cosetop(bn+1,sn)


def mapping(subgroup,subop,num):
    map01=np.zeros(num,dtype=int)
    minbrutal=np.inf
    sp=0
    numh=int(2**np.ceil(np.log2(num)/2))
    numt=int(2**np.ceil(np.log2(num)))
    k=0
    m=numt-numh
    map01[0]=m
    def search():
        for i in range(k+1,num):
            tmp=[k,i]
            for j in range(len(subgroup)):
                if (tmp in subgroup[j]) and (map01[i]==0):
                    if int(map01[k])^subop[j] not in map01:
                        map01[i]=int(map01[k])^subop[j]
                    break
    
    def randgive(mapnew):
        for i in range(numt):
            if i not in mapnew:
                if len(np.where(mapnew==0)[0])>1: 
                    mapnew[np.where(mapnew==0)[0][0]]=i
        return mapnew
                        
    
    def brutaltest(map01):
        brutals=0
        for l in range(len(subgroup)):
            for i in subgroup[l]:
                if map01[i[0]]^map01[i[1]]!=subop[l]:
                    brutals+=1
        return brutals
                    
    while sum(map01==0)>1:
        search()
        k+=1
        if(k>num):
            tmp=len(np.where(map01==0)[0])
            mapnew=randgive(map01)
            brutals=brutaltest(mapnew)
            if brutals<minbrutal:
                minbrutal=brutals
                mapmax=mapnew
                spo=tmp
            m+=1
            if m>=numt:
                map01=mapmax
                sp=spo
                break
            else:
                map01=np.zeros(num,dtype=int) 
                k=0
                map01[0]=m
        
    return map01,minbrutal,sp

def brutal(s1,s2,c1,c2):
    #s1,s2 new state c1,c2 old state in 10 form
    tmp=''
    binop=''
    for i in range(len(s1)):
        k1=s1[i]
        k2=s2[i]
        pt=c1^c2
        A="{0:b}".format(pt)
        for s in range(1,len(A)):
            if A[-s]=='1':
                break
            else:
                pt=pt^2**(s-1)
                        
        parity=(-1)**((bin(c1&((2**(len(bin(pt))-2)-1)^pt)).count('1'))%2)
        if(pt==0):
            parity=1
        if (k1=='0' and k2=='0') or (k1=='1' and k2=='0'):
            tmp+='0'
        elif (k1=='1' and k2=='1') or (k1=='0' and k2=='1'):
            tmp+='1'
        if (k1=='0' and k2=='0') or (k1=='1' and k2=='1'):
            binop+='0'
        elif (k1=='1' and k2=='0') or (k1=='0' and k2=='1'):
            binop+='1'
 
    sign=np.complex64(parity*maker(tmp)/2**len(s1))
    op=[]
    for i in range(2**len(s1)):
        tmp=''
        icount=0
        itmp="{0:b}".format(i)
        itmp=itmp.zfill(len(s1))
        for j in range(len(binop)):
            j1=itmp[j]
            j2=binop[j]
            if j1=='0' and j2=='0':
                tmp+='I'
            elif j1=='1' and j2=='0':
                tmp+='Z'
            elif j1=='0' and j2=='1':
                tmp+='X'
            else:
                icount+=1
                tmp+='Y'
                
        if icount%2==0 and icount!=0:
            if (-1)**(icount//2)<0:
                tmp='-'+tmp
        elif icount%2==1:
            tmp='i'+tmp
            if (-1)**(icount//2)<0:
                tmp='-'+tmp
        op.append(tmp)
    
    for k in range(len(sign)):
        if op[k][0]=='-':
            sign[k]=-sign[k]
            op[k]=op[k][1:]
        if op[k][0]=='i':
            sign[k]=1j*np.array(sign[k])
            op[k]=op[k][1:]
            
    return sign,op
        
    

def signmap(C2, subgroup, CV,C,subop,MAPC):
    sign=[]
    spsign=[]
    for l in range(len(subgroup)):
        signtmp=np.array([0]*(2**len(C2[0])),dtype='float64')
        sptmp=[]
        for i in subgroup[l]:
            tmp=''
            if MAPC[i[0]]^MAPC[i[1]]!=subop[l]:
                a,b=brutal(C2[i[0]],C2[i[1]],CV[i[0]],CV[i[1]])
                sptmp.append(WeightedPauliOperator([[c, label2Pauli(p)] for (c,p) in zip(a, b)]))
            else:
                for j in range(len(C2[i[0]])):
                    k1=C2[i[0]][j]
                    k2=C2[i[1]][j]
                    pt=CV[i[0]]^CV[i[1]]
                    A="{0:b}".format(pt)
                    for s in range(1,len(A)):
                        if A[-s]=='1':
                            break
                        else:
                            pt=pt^2**(s-1)
                            
                    parity=(-1)**((bin(CV[i[0]]&((2**(len(bin(pt))-2)-1)^pt)).count('1'))%2)
                    if(pt==0):
                        parity=1
                    if (k1=='0' and k2=='0') or (k1=='1' and k2=='0'):
                        tmp+='0'
                    elif (k1=='1' and k2=='1') or (k1=='0' and k2=='1'):
                        tmp+='1'
                        
                signtmp+=parity*maker(tmp)/2**len(C2[0])
        sign.append(signtmp)
        spsigntmp=WeightedPauliOperator([])
        for i in sptmp:
            spsigntmp+=i
        spsign.append(spsigntmp)
    return sign,spsign
"""
A:I+Z X+iY
B:I-Z X-iY
(X+Y)(X-Y)(I+Z)(I-Z)
->
ABAB
D=A(x)B(x)A(x)B
"""
def maker(k):
    A=[1,1]
    B=[1,-1]
    D=[1]
    for i in k:
        if i=='0':
            D=np.kron(D,A)
        elif i=='1':
            D=np.kron(D,B)    
    return D

"""
(X+Y)(X-Y)(I+Z)(I-Z)
->
QQNN
D=Q(0,1)(x)Q(0,1)(x)N(0,1)(x)N(0,1)
opmaker(sign_matrix,where is Q+, where is Q-, number of qubit)
"""
def opmaker(sign,subop,n):
    result=[]
    for i in range(len(subop)):
        E=[]
        signtmp=np.where(sign[i])
        # signtmp=np.arange(2**n)
        binop="{0:b}".format(subop[i])
        binop=binop.zfill(n)
        for k in signtmp[0]:
            bins="{0:b}".format(k)
            bins=bins.zfill(n)
            tmp=''
            icount=0
            for j in range(len(binop)):
                j1=binop[j]
                j2=bins[j]
                if j1=='0' and j2=='0':
                    tmp+='I'
                elif j1=='0' and j2=='1':
                    tmp+='Z'
                elif j1=='1' and j2=='0':
                    tmp+='X'
                else:
                    icount+=1
                    tmp+='Y'
        # i*i=-1 only i print i
            if icount%2==0 and icount!=0:
                if (-1)**(icount//2)<0:
                    tmp='-'+tmp
            elif icount%2==1:
                tmp='i'+tmp
                if (-1)**(icount//2)<0:
                    tmp='-'+tmp
            E.append(tmp)   
        result.append(np.array(E))
    return result

def getmap(n,m):
    """
    RHF Spin up= Spin down
    """
    C=[]
    fer={}
    n1=int(np.ceil(n/2))
    n2=int(np.floor(n/2))
    m1=int(np.ceil(m/2))
    m2=int(np.floor(m/2))
    x1=[]
    x2=[]
    for b in bitmasks(n1,m1):
        x1.append(b.to01())
        
    for b in bitmasks(n2,m2):
        x2.append(b.to01())
        
    for i in x1:
        for j in x2:
            C.append(i+j)
    """
    UHF FULL Configuration
    """
    # x1=[]
    # for b in bitmasks(n,m):
    #     x1.append(b.to01())
        
    # for i in x:
    #     C.append(i)
            
    numbit=int(np.ceil(np.log2(len(C))))
    print('Initializing:')
    t1=time.time()
    CV=to10(C,n)
    GOP=gop(n)
    subgroup=coset(CV,GOP)
    t2=time.time()
    print('time cost',t2-t1)
    print('guess qubits:',numbit)
    t1=time.time()
    subop,numbit=cosetop(numbit,n)
    MAPC,sp,numrand=mapping(subgroup,subop,len(C))
    t2=time.time()
    print('group preserving qubits:',numbit,'number of random given',numrand,'time cost',t2-t1)
    C2=to2(MAPC,numbit)
    sign,spsign=signmap(C2, subgroup, CV,C,subop,MAPC)
    op=opmaker(sign,subop,numbit)
    for i in range(len(sign)):
        sign[i]=np.complex64(sign[i][sign[i]!=0])

    num=0
    for i in range(n):
        for j in range(i+1,n):
            key= (j,i)
            for k in range(len(sign[num])):
                if op[num][k][0]=='-':
                    sign[num][k]=-sign[num][k]
                    op[num][k]=op[num][k][1:]
                if op[num][k][0]=='i':
                    sign[num][k]=1j*sign[num][k]
                    op[num][k]=op[num][k][1:]
            fer[key]=WeightedPauliOperator([[c, label2Pauli(p)] for (c,p) in zip(sign[num], op[num])])
            fer[key]+=spsign[num]
            num+=1

    subgroup0=[]
    for i in range(n):
        tmp=np.where(CV==CV|2**i)[0]
        stmp=[]
        for j in tmp:
            stmp.append([j,j])
        subgroup0.append(stmp)
        
    subop0=np.array([0]*n)
    sign0,spsign0=signmap(C2, subgroup0, CV,C,subop0,MAPC)
    op0=opmaker(sign0,subop0,numbit)
    for i in range(len(sign0)):
        sign0[i]=sign0[i][sign0[i]!=0]
    
    for i in range(n):
        key= (i,i)
        fer[key]=WeightedPauliOperator([[c, label2Pauli(p)] for (c,p) in zip(sign0[i], op0[i])])
        
    fer=complete_mapping(fer)

    return fer
# n=10
# m=5
# fer=getmap(n,m)
# for i in range(n-1):
#     print(np.where(abs(sum(fer[0,i].to_opflow().to_matrix()))>1))
# state=8
# electron=2
# ####       
# fer=getmap(state,electron)
