import numpy as np
class KNN:
    def __init__(self, k) : 
        self.k = k
        self.testSet = []
        self.classes = []
        self.distances = []
    
    def fit(self,testSet,classes) :
        self.testSet = testSet
        self.classes = classes
    
    #A function that takes a point and returns the k-nearest-neighbors to it.
    def takeKnn(self,point) : 
        
        #Calculate distance from point to all testSamples.
        for sample in self.testSet :
           self.distances.append(self.manhatten(point,sample))
        
        #Merge testset, classes and distances
        X = []
        for i in range(len(self.testSet)) :
            X.append([self.testSet[i],self.classes[i],self.distances[i]])

        #Bruteforcing knearest by sorting entire list and taking the k-first in list.
        X = self.sortByDist(X)
        
        resultNeighbors = []
        for i in range(self.k) :
            resultNeighbors.append(X[i])
        
        return resultNeighbors
        
    # A function that sorts a lyst by the distance_index
    def sortByDist(self,X) :
        from operator import itemgetter
        return sorted(X, key=itemgetter(2))

    # A function that calculates the manhatten distance between to points
    def manhatten(self,A,B):
        
        distance = 0

        for dim in range(len(A)):
            distance += abs(int(A[dim])-int(B[dim]))
        return distance

    # A function that returns a k-size slice of a list.
    def takeKNN(X):
        KNN = []
        for i in range(self.k):
            KNN.append(L[i])
        return KNN

    # A function that counts number of each class, and returns the class with the most counts.
    def predictClass(self, KNN):
       
       #count classes and save result in dict
        classes = {}
        for el in KNN:
            if el[1][0] not in classes:
                classes[el[1][0]] = 1
            else:
                classes[el[1][0]] += 1
        
        # Go through results and return the highest class by count
        result = ""
        highest = 0
        for key,value in classes.items():
            if value > highest:
                result = key
                highest = value
        
        return result
    
########################### USAGE AREA #####################################################


# Load test samples
L = np.array([[1,3,"Circle"],\
    [1,8,"Circle"],\
    [1,9,"Circle"],\
    [4,6,"Circle"],\
    [5,7,"Circle"],\
    [6,8,"Circle"],\
    [7,6,"Circle"],\
    [5,4,"Square"],\
    [6,1,"Square"],\
    [6,3,"Square"],\
    [7,2,"Square"],\
    [7,4,"Square"],\
    [8,2,"Square"],\
    [8,3,"Square"]])


#prepare data, we want points and classes seperate
testSamples = L[:,:2]
classes = L[:,2:3]

#Make K-nearest-neighbor object : k = number of neighbors
neighbors = KNN(4)

#Fit classes to testsamples
neighbors.fit(testSamples,classes)

# Get the k-nearest neibors
point_in = [6,6]
knn = neighbors.takeKnn(point_in)

# Predict which class the point has
predictedClass = neighbors.predictClass(knn)

#print the results nicely
print("")
print(f'-------- The nearest neighbors to the point{point_in} -------- ')
print("")
for point in knn:
    print(f"Point: {point[0]} Class: {point[1]} Distance: {point[2]}")
print("")
print(f"The given point {point_in} has class: {predictedClass}")