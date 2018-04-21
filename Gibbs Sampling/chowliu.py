import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import pickle

from collections import defaultdict

import math
import heapq

import networkx as nx


def ProcessPQ(joints, marg, feature_length):
  """
  Populates a heap in descending order of mutual informations.
  This is used to build the maximum spanning tree.
  Contains mutual information of every feature with respect to every other feature
  """
  #variable defining the heap
  pq = []

  for i in range(feature_length):
    for j in range(i+1, feature_length):
      I = 0
      for x_u, p_x_u in marg[i].iteritems():
        for x_v, p_x_v in marg[j].iteritems():
          if (x_u, x_v) in joints[(i, j)]:
            p_x_uv = joints[(i, j)][(x_u, x_v)]
            I += p_x_uv * (math.log(p_x_uv, 2) - math.log(p_x_u, 2) - math.log(p_x_v, 2))
      heapq.heappush(pq, (-I, i, j))

  return pq


def findSet(parent, i):

  while i != parent[i]:
    i = parent[i]

  return i

def buildMST(pq, feature_length):
  """
  Builds the MST using the heap generated above.
  It returns the edges that needs to be connected using the highest mutual information
  """

  parent = range(feature_length)
  size = [1]*feature_length

  count = 0
  edges = set()
  while count < feature_length-1:
    item = heapq.heappop(pq)
    i = item[1]
    j = item[2]
    seti = findSet(parent, i)
    setj = findSet(parent, j)
    if seti != setj:
      if size[seti] < size[setj]:
        size[setj] += size[seti]
        parent[seti] = setj
      else:
        size[seti] += size[setj]
        parent[setj] = seti
      edges.add((i, j))
      count += 1

  return edges

G2 = None
pos2 = None

def buildVisual(edges, feature_length, labels, fname, title=None):

  """
  Tree built could be visualized.
  This is just for visual perspectives.
  Saves the graph
  """

  global G2
  global pos2

  if type(G2) == type(None):
    G = nx.DiGraph()
    for i in range(feature_length):
      G.add_node(i)
    pos = nx.spring_layout(G, k=10., scale = 10)
    G2 = G
    pos2 = pos
  else:
    G = G2
    pos = pos2

  nx.draw_networkx_nodes(G, pos, node_size=1000)

  nx.draw_networkx_labels(G, pos,labels,font_size=8)
  nx.draw_networkx_edges(G, pos, edgelist=list(edges), arrows = True)
  if title:
    plt.title(title)
  plt.savefig(fname)
  plt.close()


##############main#####################

labels = {0: "Age",
          1: "Workclass",
          2: "education",
          3: "education-num",
          4: "marital-status",
          5: "occupation",
          6: "relationships",
          7: "race",
          8: "sex",
          9: "capital-gain",
          10: "capital-loss",
          11: "hours-per-week",
          12: "native-country",
          13: "salary",
         }

f = open("data.txt", "r")
joints = {}
marg = {}
trij = {}

feature_length = 14
data_size = 25000

for i in range(feature_length):
  marg[i] = defaultdict(float)

  for j in range(i+1, feature_length):
    joints[(i, j)] = defaultdict(float)
      
    for k in range(j+1, feature_length):
      trij[(i, j, k)] = defaultdict(float)

count_aggr = 0

#Reading of file
for line in f:
  n = line.strip().split("  ")
  count_aggr += 1

  #Calculates the marginal and joint distributions of the dataset
  #Each marginal feature has a dictionary telling the probability of getting that value
  #For each of the pair of features what is the probability of getting the two value pair together
  for i in range(feature_length):
    marg[i][n[i]] += 1./data_size

    for j in range(i+1, feature_length):
      joints[(i,j)][(n[i], n[j])] += 1./data_size
      
      for k in range(j+1, feature_length):
        trij[(i, j, k)][(n[i], n[j], n[k])] += 1./data_size 
f.close()
#Reading end

pq = ProcessPQ(joints, marg, feature_length)
edges = buildMST(pq, feature_length)
buildVisual(edges, feature_length, labels, "final.jpg", title="%d samples"%data_size)

#Consider p is parent of q
#You can use the joint probabilities and marginal probabilities calculated above in your code
#Write your code here

print edges

test_samples =  []

f = open("data_test.txt")
for line in f:
	data = line.strip().split("  ")
	test_samples.append(data)    #do all the data reading
f.close()

#save the Network after running once
fl = open('Bayes_Net', 'wb')
pickle.dump(marg, fl)
pickle.dump(joints, fl)
pickle.dump(trij,fl)
pickle.dump(test_samples, fl)
fl.close()
