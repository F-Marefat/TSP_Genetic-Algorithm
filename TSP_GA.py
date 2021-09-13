import numpy as np, random, operator, pandas as pd
import matplotlib.pyplot as plt
import math,copy,os,time
import csv
try:
    from tkinter import *
    from tkinter.ttk import *
except Exception as e:
    print("[ERROR]: {0}".format(e))
    from Tkinter import *

###################
# GLobal variables
##################
popSize=100
eliteSize=20
mutationRate=0.01
generations=100


class City:
    counter = 0
    def __init__(self,name,x,y):
        self.name = name
        self.x = self.graphX = x
        self.y = self.graphY = y
        City.counter += 1
        self.label= City.counter

    def distance(self,city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis**2)+(yDis**2))
        return distance

    def __repr__(self):
        return "label = "+ str(self.label)+"   "+"(" + str(self.x) + "," + str(self.y) + ")"

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialePopulation(popSize,cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(cityList):
    fitnessResults = {}
    for i in range(0,len(cityList)):
        fitnessResults[i] = Fitness(cityList[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(cityList, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(cityList[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedcityList(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatecityList(cityList, mutationRate):
    mutatedPop = []

    for ind in range(0, len(cityList)):
        mutatedInd = mutate(cityList[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedcityList(matingpool, eliteSize)
    nextGeneration = mutatecityList(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(cityList, popSize, eliteSize, mutationRate, generations):
    pop = initialePopulation(popSize, cityList)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def createCityListRandom():
    cityList = []
    for i in range(0, 25):
        cityList.append(City(str(i),x=int(random.random() * 200), y=int(random.random() * 200)))
    return cityList

def createCityListCircle():
    cityList = []
    for i in range(0,30):
        angle = random.uniform(0,1)*(math.pi*2)
        x=math.cos(angle)
        y=math.sin(angle)
        cityList.append(City(i,x,y))
    # plotCircle(cityList)
    return cityList

def createCityListC():
    cityList = []
    for i in range(0,20):
        angle = random.uniform(0.25,0.75)*(math.pi*2)
        x=math.cos(angle)*2
        y=math.sin(angle)*2
        cityList.append(City(str(i),x,y))
    for i in range(20,40):
        angle = random.uniform(0.25,0.75)*(math.pi*2)
        x=math.cos(angle)*3
        y=math.sin(angle)*3
        cityList.append(City(str(i),x,y))

    # plotCircle(cityList)
    return cityList

def geneticAlgorithmPlot(cityList, popSize, eliteSize, mutationRate, generations):
    pop = initialePopulation(popSize, cityList)
    progressBest = []
    progressWorst = []
    progressMean = []
    progressBest.append(1 / rankRoutes(pop)[0][1])
    progressWorst.append(1 / rankRoutes(pop)[popSize-1][1])
    sum = 0
    for i in range(0,popSize):
        sum += 1/rankRoutes(pop)[i][1]
    mean = sum / popSize
    progressMean.append(mean)

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progressBest.append(1 / rankRoutes(pop)[0][1])
        progressWorst.append(1 / rankRoutes(pop)[popSize - 1][1])
        sum = 0
        for i in range(0, popSize):
            sum += 1 / rankRoutes(pop)[i][1]
        mean = sum / popSize
        progressMean.append(mean)

    plt.plot(progressBest)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.plot(progressWorst)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.plot(progressMean)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    for i in range(0,len(bestRoute)):
        print(bestRoute[i])

    return bestRoute

def plotCircle(pop):
    x=[]
    y=[]
    l=[]
    for i in range(0,len(pop)):
        x.append(pop[i].x)
        y.append(pop[i].y)
        print(pop[i].label)
        print ("("+str(pop[i].x)+","+str(pop[i].y)+")")

    plt.scatter(x,y)
    plt.show()

pop = geneticAlgorithmPlot(cityList=createCityListC(), popSize=100, eliteSize=20, mutationRate=0.01, generations=5)
# best_route = geneticAlgorithm(cityList=createCityListC(), popSize=10, eliteSize=2, mutationRate=0.01, generations=10)
# displaySolution(createCityListC(),best_route)