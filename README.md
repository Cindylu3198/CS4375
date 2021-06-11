# CS4375
#Cynthia Luna
#CXL200021
#CS 4375.0U2

#This method calculates the entropy
def calc_entropy(data):
 #calculates length of data
 entries = len(data) 
 labels = {}
 #Reads class labels from file to object "labels"
 for rec in data:
   label = rec[-1] 
   if label not in labels.keys():
     labels[label] = 0
     labels[label] += 1
 #Entropy set to zero
 entropy = 0.0
 #Calculate the probability p(x) for every class (x)
 for key in labels:
   prob = float(labels[key])/entries
 #Formula to calculate entropy
   entropy -= prob * log(prob,2) 
 #print "Entropy -- ",entropy
 #Returns the entropy
 return entropy

#Function determines best attribute for split criteria
def attribute_selection(data):
 #Gets the number of features 
 features = len(data[0]) - 1
 #Function to calculate total entropy of the data-set
 baseEntropy = calc_entropy(data)
 #Set the info-gain (information gain) to zero
 max_InfoGain = 0.0;
 bestAttr = -1
 for i in range(features):
   #Store the values of the features
   AttrList = [rec[i] for rec in data]
   #Get the unique values
   uniqueValues = set(AttrList)
   #Set entropy and attribute entropy to zero
   newEntropy = 0.0
   attrEntropy = 0.0 
   #Go through unique values and perform split
   for value in uniqueValues:
        newData = dataset_split(data, i, value) 
        #Calculate probability
        prob = len(newData)/float(len(data)) 
        #Calculate entropy for attributes
        newEntropy = prob * calc_entropy(newData) 
        attrEntropy += newEntropy 
    #Calculate information gain
    infoGain = baseEntropy - attrEntropy 
    #Find attribute with largest information gain
    if (infoGain > max_InfoGain):
    max_InfoGain = infoGain
    bestAttr = i 
 #Return attribute with max information gain
 return bestAttr

#This function will split based on attribute with max info gain
def dataset_split(data, arc, val):
 #List to store split data-set
 newData = []
 #Go through data and split it
 for rec in data: 
   if rec[arc] == val:
     reducedSet = list(rec[:arc]) 
     reducedSet.extend(rec[arc+1:])
     newData.append(reducedSet)
 #return the new list that was split
 return newData

#Build decision tree
def decision_tree(data, labels):
 #list to store class labels
 classList = [rec[-1] for rec in data]
 if classList.count(classList[0]) == len(classList):
   return classList[0]
 #Call to get attribute for split that has max info gain
 maxGainNode = attribute_selection(data)
 treeLabel = labels[maxGainNode] 
 #Represent nodes
 theTree = {treeLabel:{}}
 del(labels[maxGainNode])
 #Get the unique values
 nodeValues = [rec[maxGainNode] for rec in data]
 uniqueValues = set(nodeValues)
 for value in uniqueValues:
   subLabels = labels[:]
 #Update node values
   theTree[treeLabel][value] = decision_tree(dataset_split(data, maxGainNode, value),subLabels) 
 #Return tree
 return theTree
