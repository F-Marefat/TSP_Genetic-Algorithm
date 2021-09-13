import tkinter
import tkinter.ttk as ttk
import numpy as np, random, operator, pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import math,copy,os,time
import csv
try:
    from tkinter import *
    from tkinter.ttk import *
except Exception as e:
    print("[ERROR]: {0}".format(e))
    from Tkinter import *


class City:
    counter = 0
    def __init__(self,name,x,y):
        # graphX : graph coordinate to show the city on canvas in gui
        name = name
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

def roulettewheel_Selection(popRanked, eliteSize):
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

def rankBased_Selection(popRanked, eliteSize):
    selectionResults = []
    selected = {}
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = int(len(popRanked)*random.random())
        for i in range(0, len(popRanked)):
            if pick <= (len(popRanked)-i):
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def tournament_Selection(popRanked, eliteSize,t):
    selectionResults = []
    selected = {}
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        for j in range(0, t):
            pick = int(len(popRanked) * random.random())
            selected[j] = popRanked[pick][0]
        max = 0
        for k in range(0, t):
            if max < popRanked[selected[j]][1]:
                max = popRanked[selected[j]][0]
        selectionResults.append(popRanked[max][0])

    return selectionResults

def truncation_Selection(popRanked, eliteSize,T):
    selectionResults = []
    selected = []
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0 , int(T *(len(popRanked)))):
        selected.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = int(len(selected) * random.random())
        selectionResults.append(selected[pick])

    return selectionResults

def random_selection(popRanked, eliteSize):
    selectionResults = []
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = int(len(popRanked) * random.random())
        selectionResults.append(popRanked[pick][0])
    return selectionResults

def matingPool(cityList, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(cityList[index])
    return matingpool

# ordered recombination
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

def breedcityList(matingpool, eliteSize,crossoverRate):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        if (random.random() < crossoverRate):
            child = breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        else:
            children.append(pool[i])

    return children

def swap_mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def insert_mutate(individual, mutationRate):
    mutate = []
    if (random.random() < mutationRate):
        indexOfbit1 = int(random.random() * len(individual))
        indexOfbit2 = int(random.random() * len(individual))
        maxindex = max(indexOfbit1,indexOfbit2)
        minindex = min(indexOfbit1,indexOfbit2)

        # case: 2 random index be the same
        if maxindex != minindex:
            for i in range(0,len(individual)):
                if i < minindex:
                    mutate.append(individual[i])
                elif i == minindex:
                    mutate.append(individual[minindex])
                    mutate.append(individual[maxindex])
                elif i < maxindex:
                    mutate.append(individual[i])
                elif i == maxindex:
                    # case: maxindex = last bit
                    if(maxindex == len(individual)-1):
                        break;
                    mutate.append(individual[i+1])
                elif i> maxindex+1:
                    mutate.append(individual[i])
        else:
            return individual
    else:
        return individual
    return mutate

def inversion_mutate(individual, mutationRate):
    mutate = []
    if (random.random() < mutationRate):
        startbit = int(random.random() * len(individual))
        endbit = int(random.random() * len(individual))
        maxindex = max(startbit, endbit)
        minindex = min(startbit, endbit)
        print(str(maxindex),str(minindex))
        if maxindex != minindex:
            for j in range(0,minindex):
                mutate.append(individual[j])
            a = 0
            for i in range(minindex,maxindex+1):
                mutate.append(individual[maxindex-a])
                a = a+1
            for z in range(maxindex+1,len(individual)):
                mutate.append(individual[z])
        else:
            return individual
    else:
        return individual
    return mutate

def mutatecityList(cityList, mutationRate,choosemutation):
    mutatedPop = []

    for ind in range(0, len(cityList)):
        if choosemutation=="swap":
            mutatedInd = swap_mutate(cityList[ind], mutationRate)
            mutatedPop.append(mutatedInd)
        elif choosemutation=="inversion":
            mutatedInd = inversion_mutate(cityList[ind], mutationRate)
            mutatedPop.append(mutatedInd)
        elif choosemutation=="insert":
            mutatedInd = insert_mutate(cityList[ind], mutationRate)
            mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate,crossoverRate,chooseSelection,choosemutation):
    popRanked = rankRoutes(currentGen)
    selectionResults = []
    if chooseSelection=="Random":
        selectionResults = random_selection(popRanked,eliteSize)
    elif chooseSelection=="roulettewheel":
        selectionResults = roulettewheel_Selection(popRanked,eliteSize)
    elif chooseSelection =="tournamenet":
        selectionResults = tournament_Selection(popRanked,eliteSize,3)
    elif chooseSelection == "truncation":
        selectionResults = truncation_Selection(popRanked,eliteSize,0.5)
    elif chooseSelection =="rankbased":
        selectionResults = rankBased_Selection(popRanked,eliteSize)

    matingpool = matingPool(currentGen, selectionResults)
    children = breedcityList(matingpool, eliteSize,crossoverRate)
    nextGeneration = mutatecityList(children, mutationRate,choosemutation)
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

def GUI():

    # set graph coordinate of cities
    def set_city_gcoords(cityList):
        # defines some variables (we will set them next)
        min_x = 100000
        max_x = -100000
        min_y = 100000
        max_y = -100000
        # finds the proper maximum/minimum
        for city in cityList:
            if city.x < min_x:
                min_x = city.x
            if city.x > max_x:
                max_x = city.x
            if city.y < min_y:
                min_y = city.y
            if city.y > max_y:
                max_y = city.y
        # shifts the x so the leftmost city starts at x=0, same for y.
        for city in cityList:
            city.graphX = (city.x + (-1 * min_x) + 1)
            city.graphY = (city.y + (-1 * min_y))
        # resets the variables now we've made changes
        min_x = 100000
        max_x = -100000
        min_y = 100000
        max_y = -100000
        # finds the proper maximum/minimum
        for city in cityList:
            if city.graphX < min_x:
                min_x = city.graphX
            if city.graphX > max_x:
                max_x = city.graphX
            if city.graphY < min_y:
                min_y = city.graphY
            if city.graphY > max_y:
                max_y = city.graphY
        # if x is the longer dimension, set the stretch factor to 300 (px) / max_x. Else do it for y. This conserves aspect ratio.
        if max_x > max_y:
            stretch = 300 / max_x
        else:
            stretch = 300 / max_y
        # stretch all the cities so that the city with the highest coordinates has both x and y < 300
        for city in cityList:
            city.graphX *= stretch
            city.graphY = 300 - (city.graphY * stretch)

    def update_canvas(the_canvas, the_route, color):
        # deletes all current items with tag 'path'
        the_canvas.delete('path')
        # loops through the route
        for i in range(len(the_route)):
            # similar to i+1 but will loop around at the end
            next_i = i - len(the_route) + 1

            # creates the line from city to city
            the_canvas.create_line(the_route[i].graphX,
                                   the_route[i].graphY,
                                   the_route[next_i].graphX,
                                   the_route[next_i].graphY,
                                   tags=("path"),
                                   fill=color)
            the_canvas.pack()
            the_canvas.update_idletasks()

    def GA_loop(cityList,generations,popSize,eliteSize,mutationRate,crossoverRate,chooseselection,choosemutation,graph):
        '''
        Main logic loop for the GA. Creates and manages populations, running variables etc.
        '''

        # first graph=true so else runs and cities in canvas creates
        # need to run and locate cities before running algorithms so variable count and while is used
        count = 2
        while count>0:
            if graph==0:
                # takes the time to measure the elapsed time
                count = 0
                start_time = time.time()

                # Creates the population:
                print("Creates the population:")
                pop = initialePopulation(popSize, cityList)
                print("Finished Creation of the population")

                # plot progress
                progressBest = []
                progressWorst = []
                progressBest.append(1 / rankRoutes(pop)[0][1])
                progressWorst.append(1 / rankRoutes(pop)[popSize - 1][1])

                # fittest used in drawing canvas
                bestRouteIndex = rankRoutes(pop)[0][0]
                popFittest = pop[bestRouteIndex]

                # gets the best length from the first population (no practical use, just out of interest to see improvements)
                initial_length = Fitness(popFittest).routeDistance()

                # Creates a random route called best_route. It will store our overall best route.
                best_route = createRoute(cityList)

                # update_canvas(canvas_current, popFittest, 'red')
                update_canvas(canvas_current, best_route, 'green')

                # Main process loop (for number of generations)
                for i in range(1, generations):
                    # Updates the current canvas every n generations (to avoid it lagging out, increase n)
                    if i % 8 == 0:
                        # update_canvas(canvas_current, best_route, 'red')
                        update_canvas(canvas_current, best_route, 'green')

                    # Evolves the population:
                    pop = nextGeneration(pop, eliteSize, mutationRate,crossoverRate,chooseselection,choosemutation)
                    bestRouteIndex = rankRoutes(pop)[0][0]
                    bestRoute = pop[bestRouteIndex]

                    # If we have found a new shorter route, save it to best_route
                    if Fitness(bestRoute).routeDistance() < Fitness(best_route).routeDistance():
                        # set the route (copy.deepcopy because pop.fittest is persistent in this loop so will cause reference bugs)
                        best_route = copy.deepcopy(bestRoute)

                    progressBest.append(1 / rankRoutes(pop)[0][1])
                    progressWorst.append(1 / rankRoutes(pop)[popSize - 1][1])

                # takes the end time of the run:
                end_time = time.time()
                # plot progress
                fig = plt.figure(1,figsize=(5.3,4))
                plt.ion()
                plt.plot(progressBest)
                plt.ylabel('Distance')
                plt.xlabel('Generation')
                plt.plot(progressWorst)
                plt.ylabel('Distance')
                plt.xlabel('Generation')
                canvas = FigureCanvasTkAgg(fig, frame4)
                canvas.get_tk_widget().grid(column=0,row=1)
                canvas.draw()

                for i in range(0, len(bestRoute)):
                    print(bestRoute[i])

                # Prints final output to terminal:
                print('Finished evolving {0} generations.'.format(generations))
                print("Elapsed time was {0:.1f} seconds.".format(end_time - start_time))
                print(' ')
                print('Initial best distance: {0:.2f}'.format(initial_length))
                print('Final best distance:   {0:.2f}'.format(Fitness(best_route).routeDistance()))
                print('The best route went via:')
                # best_route.pr_cits_in_rt(print_route=True)
            if graph==1:
                set_city_gcoords(cityList)
                graph = 0
                count = 1

    #####################
    # action listener
    #####################
    def runAlgorithm(event):
        popSize = int(textbox_pop.get())
        eliteSize = int(textbox_elit.get())
        mutationRate = float(textbox_mut.get())
        generations = int(textbox_generation.get())
        crossoverRate = float(textbox_coR.get())
        createcity = combo_city.get()
        chooseselection = combo_s.get()
        choosemutation = combo_mut.get()
        if createcity=="Random": GA_loop(createCityListRandom(),generations,popSize,eliteSize,mutationRate,crossoverRate,chooseselection,choosemutation,graph=1)
        if createcity=="Circle": GA_loop(createCityListCircle(),generations,popSize,eliteSize,mutationRate,crossoverRate, chooseselection,choosemutation,graph=1)
        if createcity =="C": GA_loop(createCityListC(),generations,popSize,eliteSize,mutationRate,crossoverRate,chooseselection,choosemutation,graph=1)
        return

    #####################
    # GUI
    #####################

    # create main window
    window = tkinter.Tk()
    window.title("TSP")
    window.geometry('900x500')
    # canvas
    frame = tkinter.Frame(window)
    frame.grid(column=0,row=0)
    # options
    frame2 = tkinter.Frame(window)
    frame2.grid(column=0,row=3)
    # options2
    frame5 = tkinter.Frame(window)
    frame5.grid(column=1, row=3)
    # labels
    frame3 = tkinter.Frame(window)
    frame3.grid(column=0, row=2)
    # labels2
    frame6 = tkinter.Frame(window)
    frame6.grid(column=1, row=2)
    # plot
    frame4 = tkinter.Frame(window)
    frame4.grid(column=1, row=0)
    # buttom
    frame7 = tkinter.Frame(window)
    frame7.grid(column=1, row=4)

    label_pop = tkinter.Label(frame3,fg="dark green",text="Population Size",width=13)
    label_pop.grid(column=0, row=0)
    textbox_pop = tkinter.Entry(frame2,width=15,)
    textbox_pop.grid(column=0, row=0)

    label_mut = tkinter.Label(frame3,fg="dark green",text="Mutation rate",width=13)
    label_mut.grid(column=1, row=0)
    textbox_mut = tkinter.Entry(frame2,width=15)
    textbox_mut.grid(column=1, row=0)

    label_coR = tkinter.Label(frame3,fg="dark green",text="Crossover rate",width=13)
    label_coR.grid(column=2, row=0)
    textbox_coR = tkinter.Entry(frame2,width=15)
    textbox_coR.grid(column=2, row=0)

    label_elit = tkinter.Label(frame3,fg="dark green",text="Elitism Size",width=13)
    label_elit.grid(column=3, row=0)
    textbox_elit = tkinter.Entry(frame2,width=15)
    textbox_elit.grid(column=3, row=0)

    label_generation = tkinter.Label(frame6,fg="dark green",text="Generation",width=13)
    label_generation.grid(column=4, row=0)
    textbox_generation = tkinter.Entry(frame5,width=15)
    textbox_generation.grid(column=4, row=0)

    label_alg = tkinter.Label(frame6, fg="dark green", text="Algorithms",width=13)
    label_alg.grid(column=5, row=0)
    combo_alg = ttk.Combobox(frame5, width=13)
    combo_alg['values'] = ("GA", "memetic")
    combo_alg.current(0)
    combo_alg.grid(column=5, row=0)

    label_mut = tkinter.Label(frame6, fg="dark green", text="Mutation",width=13)
    label_mut.grid(column=6, row=0)
    combo_mut = ttk.Combobox(frame5, width=13)
    combo_mut['values'] = ("swap", "insert", "inversion")
    combo_mut.current(0)
    combo_mut.grid(column=6, row=0)

    label_s = tkinter.Label(frame6, fg="dark green", text="Selection",width=13)
    label_s.grid(column=7, row=0)
    combo_s = ttk.Combobox(frame5, width=13)
    combo_s['values'] = ("roulettewheel", "rankbased","truncation","tournamenet","Random")
    combo_s.current(0)
    combo_s.grid(column=7, row=0)

    label_city = tkinter.Label(frame6, fg="dark green", width=13,text="CreateCity")
    label_city.grid(column=8, row=0)
    combo_city = ttk.Combobox(frame5, width=13)
    combo_city['values'] = ("Random", "Circle", "C")
    combo_city.current(1)
    combo_city.grid(column=8, row=0)

    button = tkinter.Button(frame7,text="Run",bg="white",fg="red",width=40)
    button.bind('<Button-1>',runAlgorithm)
    button.grid(column=0, row=0)


    canvas_current_title = tkinter.Label(frame, text="Best route:")
    canvas_current_title.pack()
    # initiates two canvases, one for current and one for best
    canvas_current = tkinter.Canvas(frame, height=300, width=300,bg="white")
    canvas_current.pack()

    canvas_plot_title = tkinter.Label(frame4, text="plot")
    canvas_plot_title.grid(column=0, row=0)

    # show window
    window.mainloop()

GUI()
def test():
    citylist = createCityListCircle()
    pop = initialePopulation(100, citylist)
    popRanked = rankRoutes(pop)
    selectionResults1 = roulettewheel_Selection(popRanked, 20)
    print(str(len(selectionResults1)))
    print(selectionResults1)
    selectionResults2 = tournament_Selection(popRanked, 20,2)
    print(str(len(selectionResults2)))
    print(selectionResults2)
    selectionResults3 = truncation_Selection(popRanked, 20,0.4)
    print(str(len(selectionResults3)))
    print(selectionResults3)
    selectionResults4 = rankBased_Selection(popRanked, 20)
    print(str(len(selectionResults4)))
    print(selectionResults4)
    selectionResults5 = random_selection(popRanked, 20)
    print(str(len(selectionResults5)))
    print(selectionResults5)
    return
# test()
# print(str(insert_mutate([1,2,3,4,5,6,7],1)))