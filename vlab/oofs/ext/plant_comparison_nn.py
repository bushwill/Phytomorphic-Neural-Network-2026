import os as os
import pandas as pd
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.color as color
import sys
import math
import random
import time

from collections import OrderedDict as dict
from random import randrange, uniform
from numpy.random import normal as normal
from scipy.spatial import distance
import skan
from skan import csr
from munkres import Munkres, print_matrix, make_cost_matrix, DISALLOWED
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer




global real_plant_name, size_x, parameter_number, file_path

plant_images_path = "./Original_Images/"
real_plant_name = "Plant_063-32"
plant_image_path = plant_images_path + real_plant_name
size_x = 50  #top best plants we want to know
parameter_number = 12
file_path = "./data/synthetic_images/"


def make_index(cost_):

    m = Munkres()
    indexes = m.compute(cost_)

    total = 0
    leaf_index = []
    for row, column in indexes:
        value = cost_[row][column]
        total += value
        leaf_index.append([row, column, value])

    return leaf_index


def make_matrix(ep_p, bp_p, ep_c, bp_c):

    size = max(len(ep_c), len(ep_p))
    cost_temp = np.zeros((size, size), dtype=float)

    for i in range(0, len(ep_c)):
        for j in range(0, len(ep_p)):
            cost_temp[i, j] = distance.euclidean(ep_c[i], ep_p[j]) + distance.euclidean(bp_c[i], bp_p[j])

    return cost_temp


def parse_dataframe(bin_c):
    if skan.__version__ >= '0.12.2':
        branch_data = csr.summarize(csr.Skeleton(bin_c), separator='-')
    else:
        branch_data = csr.summarize(csr.Skeleton(bin_c))

    branch_data.head()


    edges = branch_data.loc[branch_data["branch-type"] == 1]
    branches = branch_data.loc[branch_data["branch-type"] == 2]

    points = []
    length = []

    edges = edges.reset_index()
    for i in range(edges.shape[0]):
        points.append([edges["image-coord-src-0"][i], edges["image-coord-src-1"][i]])
        points.append([edges["image-coord-dst-0"][i], edges["image-coord-dst-1"][i]])
        length.append(edges["branch-distance"][i])

    branching = []
    branches = branches.reset_index()

    for i in range(branches.shape[0]):
        branching.append([branches["image-coord-src-0"][i], branches["image-coord-src-1"][i]])
        branching.append([branches["image-coord-dst-0"][i], branches["image-coord-dst-1"][i]])

    ep = []
    bp = []
    length_edge = []

    i = 0
    while i <= len(points) - 1:
        if points[i] in branching:
            bp.append(points[i])
            ep.append(points[i + 1])
            if points[i + 1][1] > points[i][1]:
                pos = 1
            else:
                pos = 0
            length_edge.append([length[int(i / 2)], pos])


        else:
            ep.append(points[i])
            bp.append(points[i + 1])
            if points[i][1] > points[i + 1][1]:
                pos = 1
            else:
                pos = 0
            length_edge.append([length[int(i / 2)], pos])

        i = i + 2


    if len(ep) == 0:
        if branch_data["image-coord-src-0"][0] < branch_data["image-coord-dst-0"][0]:
            ep.append([branch_data["image-coord-src-0"][0], branch_data["image-coord-src-1"][0]])
            bp.append([branch_data["image-coord-dst-0"][0], branch_data["image-coord-dst-1"][0]])
        else:
            bp.append([branch_data["image-coord-src-0"][0], branch_data["image-coord-src-1"][0]])
            ep.append([branch_data["image-coord-dst-0"][0], branch_data["image-coord-dst-1"][0]])

    if len(ep) > 0:
        root = [ep[len(ep) - 1][0], ep[len(ep) - 1][1]]
    else:
        root = [bp[len(ep) - 1][0], bp[len(ep) - 1][1]]

    info = sorted(zip(bp, ep, length_edge))

    if len(ep) > 1:
        ep = ep[:-1]

    return info, ep, bp, length_edge, root


def read_real_plants():

    real_ep =[]
    real_bp =[]

    for day_real in range(2,28):
        if day_real < 10:
            image_name = plant_image_path + "/topo/Day_00" + str(
                day_real) + ".png"
        else:
            image_name = plant_image_path + "/topo/Day_0" + str(
                day_real) + ".png"

        image = io.imread(image_name)
        image_gray = color.rgb2gray(color.rgba2rgb(image))
        bin = image_gray > 0.1
        info_c, ep_c, bp_c, length_c, root_c = parse_dataframe(bin)
        real_bp.append(bp_c)
        real_ep.append(ep_c)

    return real_bp, real_ep



def calculate_cost(day_syn_bp, day_syn_ep,  real_bp, real_ep):
    cost_ = make_matrix(day_syn_ep, day_syn_bp, real_ep, real_bp)
    index = make_index(cost_)
    flag = 0

    distance_cost = []
    for i in range(0, len(index)):
        real_index = index[i][0]
        syn_index = index[i][1]

        if ((real_index < len(real_ep)) & (syn_index < len(day_syn_ep))):
            distance_cost.append(
                distance.euclidean(day_syn_ep[syn_index], real_ep[real_index]) + \
                distance.euclidean(day_syn_bp[syn_index], real_bp[real_index]))
        else:
            flag = flag + 1

    while (flag > 0):
        distance_cost.append(max(distance_cost))
        flag = flag - 1

    return sum(distance_cost)



def read_syn_plants(plants):
    f = open(file_path + plants + "/output.txt", "r")
    lines = f.readlines()
    day_temp = 0
    syn_bp = []
    syn_ep = []
    index = 0
    syn_bp_day = []
    syn_ep_day = []
    day = []

    for line in lines:
        temp = line.split(" ")
        if temp[0] == "Day:":
            day_temp = int(temp[1])
            if day_temp>2:
                syn_bp.append(syn_bp_day)
                syn_ep.append(syn_ep_day)
                syn_bp_day = []
                syn_ep_day = []
        if (temp[0] != "Day:") & (day_temp > 1):
            if temp[0] == "I":
                syn_bp_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)
            else:
                syn_ep_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)

    if day_temp == 27:
        syn_bp.append(syn_bp_day)
        syn_ep.append(syn_ep_day)

    f.close()

    return syn_bp, syn_ep



def calculate_each_plant_cost(real_bp, real_ep):

    syn_plants = os.listdir(file_path)
    size = len(syn_plants)
    cost = np.zeros((size, 1), dtype=float)

    index_min_cost = np.zeros((size, 1), dtype=float)

    #print(real_bp)


    for j in range(0, len(syn_plants)):
        plant = syn_plants[j]
        syn_bp, syn_ep = read_syn_plants(plant)
        

        #start_l2 = timer()

        
        cost_plant = []
        for i in range(0, len(min([syn_bp, syn_ep, real_bp, real_ep]))):
            cost_day = calculate_cost(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
            cost_plant.append(cost_day)

        cost[j] = sum(cost_plant)


        #end = timer()
        #print(f"Runtime of the program for cost function second loop is {end - start_l2}")

        #print(plant, cost[j], index_min_cost[j])


    if cost.size > 0:
        min_index = np.argmin(cost)
        f_p = open("./plants_cost_min.txt", "a")
        f_p.write(str(syn_plants[min_index]) + " " + str(cost[min_index]) + "\n")
        f_p.close()
        
    #print(syn_plants[min_index], cost[min_index])
    
    return syn_plants, cost



def read_parameters_from_files(plant_name):
    f = open("./parameter_values.txt", "r")
    lines = f.readlines()
    flag = 0
    i = 0
    parameter_value = []

    for line in lines:
        temp = line.split()
        if ((flag == 1) & (i < 12)):
            parameter_value.append(temp[1])
            i = i + 1
        if temp[0] == plant_name:
            flag = 1

        if ((flag == 1) & (i == 12)):
            f.close()
            return parameter_value


def read_parameters(syn_plants, sorted_index):
    parameter = np.zeros((size_x, parameter_number), dtype=float)

    for i in range(0, size_x):
        plant_name = syn_plants[sorted_index[i]]
        parameter_value = read_parameters_from_files(plant_name)

        for j in range(0, parameter_number):
            parameter[i, j] = parameter_value[j]

    return parameter


def seq_crossover(parameter):
    cross_ind_max = parameter_number - 1
    i = 0
    while i < size_x - 1:

        irand = randrange(0, cross_ind_max)
        # print(i, i + 1)
        for j in range(irand + 1, parameter_number):
            temp = parameter[i, j]
            parameter[i, j] = parameter[i + 1, j]
            parameter[i + 1, j] = temp
        i = i + 2

    return parameter


def random_crossover(parameter):
    ind = random.sample(range(0, size_x), size_x)
    cross_ind_max = parameter_number - 1
    i = 0

    while i < size_x - 1:

        irand = randrange(0, cross_ind_max)
        ind_1 = ind[i]
        ind_2 = ind[i + 1]
        # print(ind_1, ind_2, irand)

        for j in range(irand + 1, parameter_number):
            temp = parameter[ind_1, j]
            parameter[ind_1, j] = parameter[ind_2, j]
            parameter[ind_2, j] = temp
        i = i + 2

    return parameter

def mutation(parameter):
    for i in range(0, size_x):
        irand = randrange(0, parameter_number)

        if irand == 0:
            parameter[i, irand] = normal(10.0, 1.0)

        if irand == 1:
            parameter[i, irand] = normal(3.0, 0.1)

        if irand == 2:
            parameter[i, irand] = normal(0.0, 4.0)

        if irand == 3:
            parameter[i, irand] = normal(135.0, 5.0)

        if irand == 4:
            parameter[i, irand] = normal(5.0, 1.0)

        if irand == 5:
            parameter[i, irand] = normal(0.5, 0.01)

        if irand == 6:
            parameter[i, irand] = normal(1.0, 0.1)

        if irand == 7:
            parameter[i, irand] = normal(90.0, 3.0)

        if irand == 8:
            parameter[i, irand] = normal(180.0, 3.0)

        if irand == 9:
            parameter[i, irand] = normal(0.7, 0.05)

        if irand == 10:
            parameter[i, irand] = normal(0.9, 0.01)

        if irand == 11:
            parameter[i, irand] = normal(0.5, 0.01)

    return parameter

