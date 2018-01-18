#make the data watermelon_2 from 3
# import numpy as np
# import pandas as pd
# dataset = pd.read_csv('/home/parker/watermelonData/watermelon_3.csv', delimiter=",")
# del dataset['密度']
# del dataset['含糖率']
# dataset.to_csv('/home/parker/watermelonData/watermelon_2.csv',header=True,index=False)

# import numpy as np
# import pandas as pd
# dataset = pd.read_csv('/home/parker/watermelonData/watermelon_2.csv', delimiter=",")
# #print(dataset)
# trainID=[0,1,2,5,6,9,13,14,15,16]
# testID=[3,4,7,8,10,11,12]
# trainData=dataset.iloc[trainID,range(8)]
# testData=dataset.iloc[testID,range(8)]
# print(trainData)
# print(testData)
# trainData.to_csv('/home/parker/watermelonData/watermelon_2train.csv', header=True, index=False)
# testData.to_csv('/home/parker/watermelonData/watermelon_2test.csv', header=True, index=False)

import numpy as np
import pandas as pd
dataset = pd.read_csv('/home/parker/watermelonData/watermelon_2train.csv', delimiter=",")
testData = pd.read_csv('/home/parker/watermelonData/watermelon_2test.csv', delimiter=",")
print(dataset)

Attributes=dataset.columns
m,n=np.shape(dataset)
# print(m,n)

dataset=np.matrix(dataset)
attributeSet=[]
for i in range(n):
    curSet=set()
    for j in range(m):
        curSet.add(dataset[j,i])
    attributeSet.append(curSet)

DD=np.arange(0,m,1)
AA=np.ones(n)
AA=list(AA)
AA[0]=AA[n-1]=-1
EPS=1

import random
import copy
def treeGenerate(D,A,title):
    node=Node(title)
    if isSameY(D):#p74 condition(1),samples are in the same cluster
        node.v=dataset[D[0], n - 1]
        return node
    if isBlankA(A) or isSameAinD(D,A):#condition(2),A==NULL or all the D have the same attribute selected
        node.v=mostCommonY(D)
        return node
    #choose the best attribute
    giniGain=n#max num=n , formula 4.6
    floatV=0
    sameValue = []  # for random choose
    for i in range(len(A)):
        if(A[i]>0):
            curGini,divideV=giniIndex(D,i)#formula 4.6
            # print(Attributes[i],curGini)
            if curGini<=giniGain:
                if curGini<giniGain:
                    sameValue=[i]
                    p=i
                    giniGain=curGini
                    floatV=divideV
                else:#random choose
                    sameValue.append(i)

    p = sameValue[random.randint(0,len(sameValue)-1)]
    # print("\n")
    if isSameValue(-1000,floatV,EPS):#not a float devide
        node.v=Attributes[p]+"=?"
        curSet=attributeSet[p]
        for i in curSet:
            Dv=[]
            for j in range(len(D)):
                if dataset[D[j],p]==i:
                    Dv.append(D[j])
            if Dv==[]:#condition(3)
                nextNode = Node(i)
                nextNode.v=mostCommonY(D)
                node.children.append(nextNode)
                #book said we should return here, but I think we should continue
            else:
                #newA=A[:]
                newA=copy.deepcopy(A)
                newA[p]=-1
                node.children.append(treeGenerate(Dv,newA,i))
    else:#is a float devide,the floatV is the boundary
        Dleft=[]
        Dright=[]
        node.v=Attributes[p]+"<="+str(floatV)+"?"
        for i in range(len(D)):
            if dataset[D[i],p]<=floatV:Dleft.append(D[i])
            else: Dright.append(D[i])
        #A[:] should be deepcopy, I found a bug here,A[:] does not work as a deepcopy
        #then I see why it is, numpy array can not use A[:] to deepcopy
        #change the A to list and A[:] can be a deepcopy
        node.children.append(treeGenerate(Dleft,A[:],"yes"))
        node.children.append(treeGenerate(Dright,A[:],"no"))
    return node

class Node(object):
    def __init__(self,title):
        self.title=title
        self.v=1
        self.children=[]
        self.deep=0#for plot
        self.ID=-1#for plot

def isSameY(D):
    curY = dataset[D[0], n - 1]
    for i in range(1, len(D)):
        if dataset[D[i],n-1]!=curY:
            return False
    return True

def isBlankA(A):
    for i in range(n):
        if A[i]>0:return False
    return True

def isSameAinD(D,A):
    for i in range(n):
        if A[i]>0:
            for j in range(1,len(D)):
                if not isSameValue(dataset[D[0],i],dataset[D[j],i],EPS):
                    return False
    return True

def isSameValue(v1,v2,EPS):
    # if type(v1)==type(dataset[0,8]):
    #     return abs(v1-v2)<EPS
    # else: return v1==v2
    return v1==v2

def mostCommonY(D):
    res=dataset[D[0],n-1]#1 or 0
    maxC = 1
    count={}
    count[res]=1
    for i in range(1,len(D)):
        curV=dataset[D[i],n-1]
        if curV not in count:
            count[curV]=1
        else:count[curV]+=1
        if count[curV]>maxC:
            maxC=count[curV]
            res=curV
    return res

def gini(D):#formula 4.5
    types = []
    count = {}
    for i in range(len(D)):
        curY = dataset[D[i], n - 1]
        if curY not in count:
            count[curY] = 1
            types.append(curY)
        else:
            count[curY] += 1
    ans = 1
    total = len(D)
    for i in range(len(types)):
        ans -= count[types[i]] / total * count[types[i]] / total
        #print(count[types[i]] / total * count[types[i]] / total)
    return ans

def giniIndex(D,p):#formula 4.6
    # if type(dataset[0,p])==type(dataset[0,8]):
    #     res,divideV=gainFloat(D,p)
    if False:
        nothing=1
    else:
        types=[]
        count={}
        for i in range(len(D)):
            a=dataset[D[i],p]
            if a not in count:
                count[a]=[D[i]]
                types.append(a)
            else:
                count[a].append(D[i])
        res=0
        for i in range(len(types)):
            res+=gini(count[types[i]])
        divideV=-1000
    return res,divideV


myDecisionTreeRoot=treeGenerate(DD,AA,"root")


def countLeaf(root,deep):
    root.deep=deep
    res=0
    if root.v=='是' or root.v=='否':
        res+=1
        return res,deep
    curdeep=deep
    for i in root.children:
        a,b=countLeaf(i,deep+1)
        res+=a
        if b>curdeep:curdeep=b
    return res,curdeep
cnt,deep=countLeaf(myDecisionTreeRoot,0)
def giveLeafID(root,ID):
    if root.v=='是' or root.v=='否':
        root.ID=ID
        #print(root.title,ID,root.deep)
        ID+=1
        return ID
    for i in root.children:
        ID=giveLeafID(i,ID)
    return ID
giveLeafID(myDecisionTreeRoot,0)

import matplotlib.pyplot as plt
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    plt.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,
                                textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
fig=plt.figure(1,facecolor='white')

import matplotlib as  mpl
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False


def dfsPlot(root):
    if root.ID==-1:
        childrenPx=[]
        meanPx=0
        for i in root.children:
            cur=dfsPlot(i)
            meanPx+=cur
            childrenPx.append(cur)
        meanPx=meanPx/len(root.children)
        c=0
        for i in root.children:
            nodetype=leafNode
            if i.ID<0:nodetype=decisionNode
            plotNode(i.v,(childrenPx[c],0.9-i.deep*0.8/deep),(meanPx,0.9-root.deep*0.8/deep),nodetype)
            plt.text((childrenPx[c]+meanPx)/2,(0.9-i.deep*0.8/deep+0.9-root.deep*0.8/deep)/2,i.title)
            c+=1
        return meanPx
    else:
        return 0.1+root.ID*0.8/(cnt-1)
rootX=dfsPlot(myDecisionTreeRoot)
plotNode(myDecisionTreeRoot.v,(rootX,0.9),(rootX,0.9),decisionNode)


testData=np.matrix(testData)

def treePredictSet(root,testSet):#return the precision
    testM,testN=np.shape(testSet)
    confusionMatrix=np.zeros((2,2))
    for i in range(testM):
        predictV=treePredictOne(root,testSet,i)
        trueV=testSet[i,testN-1]
        if predictV==trueV:
            if trueV=='否':confusionMatrix[0,0]+=1
            else: confusionMatrix[1,1]+=1
        else:
            if trueV=='否':confusionMatrix[0,1]+=1
            else:confusionMatrix[1,0]+=1
    return confusionMatrix

def treePredictOne(root,testSet,p):#not support float
    while(True):
        if root.children==[]:#ID!=-1 leafnode
            return root.v
        curAttribute=root.v
        for i in range(len(Attributes)):
            if Attributes[i] in curAttribute:#curAttribute.contain(Attributes[i]):
                curAttribute=i
                break
        title=testSet[p,curAttribute]
        for i in root.children:
            if i.title==title:
                root=i
                break

print(treePredictSet(myDecisionTreeRoot,testData))

# fuck=np.ones(3)
# fuck=list(fuck)
# fuck2=fuck[:]
# fuck2[0]=999
# print(fuck)


plt.show()