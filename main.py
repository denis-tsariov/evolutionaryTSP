import numpy as np

if __name__ == "__main__":
    f = open("ch130.tsp", "r")
    line = f.readline()
    while not (line[0].isdigit()):
        line = f.readline()

    cities = np.zeros((130, 2))
    index = 0
    while line.strip() != "EOF":
        line_list = line.split(" ")
        line_list.remove(line_list[0])
        line_list_float = np.array([float(i) for i in line_list])
        cities[index] = line_list_float
        index += 1
        line = f.readline()
    # the index of cities is off by one compared to their indecies in the file
    # print(cities)
    
