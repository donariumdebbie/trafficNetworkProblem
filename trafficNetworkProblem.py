import numpy as np
import scipy as sp
import math as ma
import sys
import time

def Dijkst(ist,isp,wei):
    # Dijkstra algorithm for shortest path in a graph
    #    ist: index of starting node
    #    isp: index of stopping node
    #    wei: weight matrix

    # exception handling (start = stop)
    if (ist == isp):
        shpath = [ist]
        return shpath

    # initialization
    N         =  len(wei)
    Inf       =  sys.maxint
    UnVisited =  np.ones(N,int)
    cost      =  np.ones(N)*1.e6
    par       = -np.ones(N,int)*Inf

    # set the source point and get its (unvisited) neighbors
    jj            = ist
    cost[jj]      = 0
    UnVisited[jj] = 0
    tmp           = UnVisited*wei[jj,:]
    ineigh        = np.array(tmp.nonzero()).flatten()
    L             = np.array(UnVisited.nonzero()).flatten().size

    # start Dijkstra algorithm
    while (L != 0):
        # step 1: update cost of unvisited neighbors,
        #         compare and (maybe) update
        for k in ineigh:
            newcost = cost[jj] + wei[jj,k]
            if ( newcost < cost[k] ):
                cost[k] = newcost
                par[k]  = jj

        # step 2: determine minimum-cost point among UnVisited
        #         vertices and make this point the new point
        icnsdr     = np.array(UnVisited.nonzero()).flatten()
        cmin,icmin = cost[icnsdr].min(0),cost[icnsdr].argmin(0)
        jj         = icnsdr[icmin]

        # step 3: update "visited"-status and determine neighbors of new point
        UnVisited[jj] = 0
        tmp           = UnVisited*wei[jj,:]
        ineigh        = np.array(tmp.nonzero()).flatten()
        L             = np.array(UnVisited.nonzero()).flatten().size

    # determine the shortest path
    shpath = [isp]
    while par[isp] != ist:
        shpath.append(par[isp])
        isp = par[isp]
    shpath.append(ist)

    return shpath[::-1]

def calcWei(RX,RY,RA,RB,RV):

    # calculate the weight matrix between the points

    n    = len(RX)
    wei = np.zeros((n,n),dtype=float)
    m    = len(RA)
    for i in range(m):
        xa = RX[RA[i]-1]
        ya = RY[RA[i]-1]
        xb = RX[RB[i]-1]
        yb = RY[RB[i]-1]
        dd = ma.sqrt((xb-xa)**2 + (yb-ya)**2)
        tt = dd/RV[i]
        wei[RA[i]-1,RB[i]-1] = tt
    return wei
    # calculate the original weight matrix between the points
    # RX : x-coordinates of the vertices
    # RY : y-coordinates of the vertices
    # RA : first column of RomeEdges file
    # RB : second column of RomeEdges file
    # RV : third column of RomeEdges file. represents the speed limit of each edges.

    n    = len(RX)
    wei = np.zeros((n,n),dtype=float)
    m    = len(RA)
    for i in range(m):
        xa = RX[RA[i]-1]
        ya = RY[RA[i]-1]
        xb = RX[RB[i]-1]
        yb = RY[RB[i]-1]
        dd = ma.sqrt((xb-xa)**2 + (yb-ya)**2)
        tt = dd/RV[i]
        wei[RA[i]-1,RB[i]-1] = tt
    return wei


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def newWei(wei,RT,xi):
    # This is a function creating new weight matrix at each iteration.
    # wei : original weight matrix
    # RT : traffic vector at specific time. represents the number of cars at each nodes at a specific time.
    # xi : parameter

    n = wei.shape[0]
    nwei = np.zeros((n,n),dtype = float)

    for i in range(n):
        for j in range(n):
            if wei[i,j] != 0:
                ci = RT[i]
                cj = RT[j]
                nwei[i,j] = wei[i,j] + xi*(ci+cj)/2.0
    return nwei


def mkTrafficMat(ist,isp,time,RX,RY,RA,RB,RV,xi,accident=False):
    # traffic is (58*time) shape matrix, and its rows represent each node and the columns represent each second. 
    # This matrix records the number of cars after each timestep iterations.
    traffic = np.zeros((58,time),dtype=int)

    # Form original weight matrix using the function calcWei.
    originalWei = calcWei(RX,RY,RA,RB,RV)

    # Make the weights of node 30 as zero
    if accident:
        originalWei[29,:] = 0
        originalWei[:,29] = 0

    traffic[12,0] = 20 

    # updatedWei = np.copy(originalWei)
    updatedWei = newWei(originalWei,traffic[:,0],xi)

    usedEdgeCheck = originalWei.copy()

    for t in range(1,time,1):
        # Movement step
        traffic[:,t] = np.copy(traffic[:,t-1])

        for nodej in range(58):

            # Only the nodes (indexes) where there exist cars at time before movement. 
            if traffic[nodej,t-1] != 0:

                # If nodej is 52, 60% of cars are left behind and 40% of cars will leave the network.
                if nodej != 51:

                    # Apply Dijkstra algorithm to find shortest optimal path
                    shpath = Dijkst(nodej,51,updatedWei)

                    # For solving question 3. Make the entry of the matrix 'usedEdgeCheck' be 0, when the node is used.
                    usedEdgeCheck[nodej,shpath[1]] = 0

                    # Calculate 70%
                    movingCars = int(round(traffic[nodej,t-1]*0.7))
                    stayingCars = int(traffic[nodej,t-1] - movingCars)

                    # update the current time line column of traffic matrix. 
                    traffic[shpath[1],t] += movingCars
                    traffic[nodej,t] -= movingCars  
                    
                else:
                    nonleavingCars = int(round(traffic[nodej,t-1]*0.6))
                    carsLeavingGraph = int(traffic[nodej,t-1] - nonleavingCars)  
                    traffic[nodej,t] -= carsLeavingGraph     


        #  Cars injected
        if 0 < t < 180 :
            traffic[12,t] = traffic[12,t] + 20 

        # Calculate new weight matrix with updated cars location
        updatedWei = newWei(originalWei,traffic[:,t],xi)

    return traffic, usedEdgeCheck


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # #      M       A         I        N       # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
    import numpy as np
    import scipy as sp
    import csv

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Importing 'RomeVertices' file and define RomeX, RomeY
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    RomeX = np.empty(0,dtype=float)
    RomeY = np.empty(0,dtype=float)
    with open('RomeVertices','r') as file:
        AAA = csv.reader(file)
        for row in AAA:
            RomeX = np.concatenate((RomeX,[float(row[1])]))
            RomeY = np.concatenate((RomeY,[float(row[2])]))
    file.close()

    # Importing 'RomeEdges' file and define RomeA, RomeB, RomeV
    RomeA = np.empty(0,dtype=int)
    RomeB = np.empty(0,dtype=int)
    RomeV = np.empty(0,dtype=float)
    with open('RomeEdges','r') as file:
        AAA = csv.reader(file)
        for row in AAA:
            RomeA = np.concatenate((RomeA,[int(row[0])]))
            RomeB = np.concatenate((RomeB,[int(row[1])]))
            RomeV = np.concatenate((RomeV,[float(row[2])]))
    file.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    nstart = 12 # St. Peter's Square
    nend = 51 # Coliseum
    time = 200
    xi_q1 = 0.01

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Question 1. Determine for each node the maximum load over the 200 iterations
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    Tmat, usedEdgeCheckq1 = mkTrafficMat(nstart,nend,time,RomeX,RomeY,RomeA,RomeB,RomeV,xi_q1)
    maxperNode = np.amax(Tmat,axis = 1)
    print "Question1 : The maximum load for each node over 200 iterations is", maxperNode

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Question 2. Which are the five most congested nodes?
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    maxperNodeDescend = sorted(maxperNode, reverse=True)
    print "Question2: max per node in descending order is:", maxperNodeDescend

    q2_fiveMostCong = np.array(sorted(range(len(maxperNode)), reverse = True, key=lambda k:maxperNode[k]))+1
    print "Question2: fiveMostCong: ", q2_fiveMostCong

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Question 3. Which edges are not utilized at all? Why?
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    unvisitedEdge = np.argwhere(usedEdgeCheckq1 > 0) + 1

 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Question 4. What flow pattern do we observed for parameter xi = 0?
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    xi_q4 = 0.0
    Tmat_q4, usedEdgeCheckq4 = mkTrafficMat(nstart,nend,time,RomeX,RomeY,RomeA,RomeB,RomeV,xi_q4)
    maxperNode_q4 = np.amax(Tmat_q4,axis = 1)
    print "Question 4: when xi=0,maximum over the iterations are: ", maxperNode_q4
    optimalOriginal = Dijkst(12,51,calcWei(RomeX,RomeY,RomeA,RomeB,RomeV))
    print "Optimal route with original weight matrix is:", optimalOriginal

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Question 5. An accident occurs at node 30, which blocks any route to or from node 30. Which nodes are now the most congested?
    # and what is their maximum load?
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    xi_q5 = 0.01
    TmatAccident,usedEdgeCheckq5 = mkTrafficMat(nstart,nend,time,RomeX,RomeY,RomeA,RomeB,RomeV,xi_q5,accident = True)

    # Maximum load when there is an accident at node 30
    q5_maxperNode = np.amax(TmatAccident,axis = 1)
    print "Question5 : When there is an accident in node 30, the maximum load for each node over 200 iterations is", q5_maxperNode

    q5_maxperNodeDescend = sorted(q5_maxperNode, reverse=True)
    print "Question5: max per node in descending order is:", q5_maxperNodeDescend[:6]

    q5_fiveMostCong = np.array(sorted(range(len(q5_maxperNode)), reverse = True, key=lambda k:q5_maxperNode[k])) + 1
    print "Question5: fiveMostCong: ", q5_fiveMostCong[:6]

    # Which nodes (beside 30) decrease the most in peak value?
    # Which nodes increase the most in peak value?

    peakdifference = q5_maxperNode - maxperNode
    peakdifference2 = list(set(q5_maxperNode - maxperNode))
    print "peakdifference :", peakdifference2

    decreaseValue = sorted(peakdifference2)[:6]

    # indices of the corresponding nodes
    decreasedNodes = []
    for d in decreaseValue:
        decreasedNodes_index = []
        for w in np.where(peakdifference == d)[0]:
            decreasedNodes_index.append( w + 1 )
        decreasedNodes.append(decreasedNodes_index)
 
    increaseValue = sorted(peakdifference2,reverse=True)[:5]

    # indices of the corresponding nodes
    increasedNodes = []
    for d in increaseValue:
        increasedNodes_index = []
        for w in np.where(peakdifference == d)[0]:
            increasedNodes_index.append( w + 1 )
        increasedNodes.append(increasedNodes_index)


    print "Question5: Nodes decrease the most in peak value after the accident except 30 are:", decreasedNodes[1:]
    print "Question5: and each of them decrease as: ", decreaseValue[1:]
    print "Question5: Nodes increase the most in peak value after the accident except 30 are:", increasedNodes
    print "Question5: and each of them increase as: ", increaseValue