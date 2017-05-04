import xlrd
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import csv
import time
import matplotlib.pyplot as plt


# -----------load the sample data------------
# filename：读入文件名的列表
# output：训练样本data_x，样本标签data_y
def load_data(filename):
    data_x = []
    data_y = []
    for j in range(len(filename)):
        table = xlrd.open_workbook(filename[j]).sheets()[0]
        for i in range(1, table.nrows):
            data_x.append(table.row_values(i))
            data_y.append(j+1)
    return np.array(data_x), np.array(data_y)


# -------Bit string representation & Population-------
# individual_size：基因个数
# population_size：种群大小
# output：初始种群
def get_population(individual_size, population_size):
    return np.array([[random.randint(0, 1) for j in range(individual_size)] for i in range(population_size)])


# ------Fitness / Score (Evaluation Function)--------
# population：种群
# example_x：训练赝本
# label_y：样本标签
# output：适应度函数值/评价函数值
def cal_fitness(population, example_x, label_y):
    fitness = []
    for j in range(len(population)):
        data_x = example_x.copy()
        for i in range(len(population[j])):
            if population[j][i] == 0:
                data_x[:, i] = 0
        model = KNeighborsClassifier()
        model.fit(data_x, label_y)
        fitness.append(metrics.accuracy_score(label_y, model.predict(data_x)))
    return np.array(fitness).reshape(len(population), 1)


# ------------sort Population by fitness------------------
# population：种群
# fitness：适应度函数值/评价函数值
# output：根据fitness降序排序之后的种群
def get_sort(population, fitness):
    population = np.column_stack((population, fitness)).tolist()
    population.sort(key=lambda pop: pop[-1], reverse=True)
    # print(population[0])
    return np.array(population)[:, :len(population[0])-1]


# ---------------Parent/Offspring-------------------
# population：种群
# parents_number：存活个体数目
# p_co：crossover的概率
# p_mut：mutation的概率
# output：Offspring
def get_offsprings(population, parents_number, p_co, p_mut):
    for i in range(parents_number+1, len(population)):
        if random.random() < p_co:
            pop = get_crossover(population[random.randint(0, parents_number-1)], population[random.randint(0, parents_number-1)])
            j = 10
            while (pop.tolist() in population.tolist()) and j:
                # flag = pop.tolist() in population.tolist()
                # print(flag)
                j -= 1
                pop = get_crossover(population[random.randint(0, parents_number-1)], population[random.randint(0, parents_number-1)])
            if j:
                population[i] = pop
        else:
            pop = get_mutation(population[random.randint(0, parents_number)], p_mut)
            j = 10
            while (pop.tolist() in population.tolist()) and j:
                # flag = pop.tolist() in population.tolist()
                # print(flag)
                j -= 1
                pop = get_mutation(population[random.randint(0, parents_number)], p_mut)
            if j:
                population[i] = pop
    return population


# ---------------Crossover--------------------------
# pop1，pop2：发生crossover的两个parents
# output：新的个体
def get_crossover(pop1, pop2):
    pop = pop1.copy()
    index = random.randint(0, len(pop)-1)  # a randomly chosen bit
    pop[index] = pop2[index]
    return pop


# ---------------Mutation-------------------------
# pop1：发生mutation的parent
# p_mut：mutation的概率
# output：新的个体
def get_mutation(pop1, p_mut):
    pop = pop1.copy()
    for i in range(len(pop)):
        if random.random() < p_mut:
            pop[i] = 1 - pop[i]
    return pop


# --------------Genetic algorithm-----------------
# population_size：种群大小
# individual_size：基因个数
# survival_rate：存活率
# p_co：crossover的概率
# p_mut：mutation的概率
# iteration_max：最大迭代次数
# output:日志文件
def genetic_algorithm(population_size, individual_size, survival_rate=0.3, p_co=0.6, p_mut=0.1, iteration_max=10):
    parents_number = int(population_size * survival_rate)
    # --------------Initialization Population---------------
    population = get_population(individual_size, population_size)
    # ---------------------log file---------------------
    filename = 'log'+time.strftime("%Y%m%d", time.localtime())+'.csv'
    cf = open(filename, 'a')
    writer = csv.writer(cf, dialect='excel')
    writer.writerow(['Bit string representation', 'Classification accuracy'])
    # ------------------Iteration---------------------
    print('--------------Start iteration----------------')
    fitness_max = []
    for i in range(iteration_max):
        fitness = cal_fitness(population, data_X, data_Y)  # Calculation the Fitness
        population = get_sort(population, fitness)  # Sort the Population by Fitness
        # ---------------log----------------
        fitness_max.append(fitness.max())
        writer.writerow([population[0], fitness.max()])
        print('Iteration:', i, '    Bit string representation:', population[0], '    Classification accuracy:', fitness.max())
        # print(population)
        # print('Iteration:', i, '    Bit string representation:', population[1])
        # print('Iteration:', i, '    Bit string representation:', population[2])
        # print('Iteration:', i, '    Bit string representation:', population[3])
        # print('Iteration:', i, '    Bit string representation:', population[4])
        # ---------------Generate the offspring----------------------------
        population = get_offsprings(population, parents_number, p_co, p_mut)
    cf.close()
    print('--------------End iteration----------------')
    print('Logs are stored in ' + filename)
    plt.plot(fitness_max)
    plt.title('Plot of Classification accuracy vs Number of iterations')
    plt.xlabel('Number of iterations')
    plt.ylabel('Classification accuracy')
    plt.show()


if __name__ == '__main__':
    # ----------------load the data------------------
    Filename = ['A.xls', 'B.xls', 'C.xls', 'D.xls', 'E.xls']
    data_X, data_Y = load_data(Filename)
    # ------------------parameter------------------
    POPULATION_SIZE = 30
    INDIVIDUAL_SIZE = 15
    SURVIVAL_RATE = 0.6
    P_co = 0.7
    P_mut = 0.3
    Iteration_max = 100
    # --------------Genetic algorithm-----------------
    genetic_algorithm(population_size=POPULATION_SIZE, individual_size=INDIVIDUAL_SIZE, survival_rate=SURVIVAL_RATE, p_co=P_co, p_mut=P_mut, iteration_max=Iteration_max)
