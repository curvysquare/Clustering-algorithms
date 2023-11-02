#Studentnumber: 201680340
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(2)
np.random.seed(2)
# the object class below creates an instance for every datapoint within the dataset.
class object:
 # the "label" attributes is its label, eg "tiger". "vector" is the 300 feature values.
#"distance attribute" is a dictionary that store the squared ecludian distance between
# itself and all the centroids. "silh_co" attribute is the sihouette coefficient for the
# datapoint and is initlized as zero. "nearest K" attribute is the nearest centroid.
# "prob" attribute is the probability of being selected during the Kmeans++ algorithm.
    def __init__(self, label, vector):
        self.label = label
        self.vector = vector
        self.distances = {}
        self.silh_co = 0
        # self.distances_squared = {}
        self.nearest_k = 0
        self.prob =0
    # the "get_distances" class method takes a list of the centroid objects as input.
    # it calculautes the squared euclidean distance between the datapoint and each centroid object.
    def get_distances(self, centroids):
        for centroid_x in centroids:
            # think have to be series/ dataframe. cant be list
            x= np.array(self.vector)
            y = np.array(centroid_x.vector)
            distance = (np.linalg.norm(x-y))
            self.distances[centroid_x.label] = distance**2
    # the "get_nearest_k" function inspects its self.distances attribute dictionary for the smallest distances
    # and saves the key of the nearest centroid.
    def get_nearest_k(self):
        self.nearest_k = min(self.distances, key=self.distances.get)
    # the "calc_prob" function takes the key of the nearest centroid. it takes the value of this key
    # which is the squared euclaidian distance and divides it by the sum squared distances of all objects.
    def calc_prob(self,sum_dist_Sq ):
        key_clust_min_dist = min(self.distances)

        self.prob = (self.distances[key_clust_min_dist])/ sum_dist_Sq
# the centroid class below is to create a centroid or equivalently a representitative of a cluster.
class centroid:
     # the "label" attributes is its arribitraily given centroid number. "vector" is the 300 feature values.
    def __init__(self, label, vector):
        self.label = label
        self.vector = vector
     # the "random initlise" function creates an aritfical datapoint within the scope of the dataset.
     # it takes the dataframe of the dataset and calculates the maximum and minimum of
     # values present in the dataframe. For each of the 300 features, a random number is chosen to construct
     # a 300 feature long vector of the aritifical point
    def random_initialise(self, df):
        min_list = df.min(numeric_only = True)
        max_list = df.max(numeric_only = True)
        df_width = (np.shape(df)[1])-1
        vec = []
        for j in range(1, df_width+1):
            n = np.random.uniform(min_list[j], max_list[j])
            vec.append(n)
        # vec = pd.Series(vec)
        vec = vec
        self.vector = vec
     # the "rand_intilaise_within_ d" function randomly chooses an actual datapoint of the dataset.
     # it transposes it and formats it from a pandas dataframe to a pandas series using the "squeeze" method
    def random_initialise_within_D(self, df):
        y1 = df.sample(n=1)
        y1 = y1.T.squeeze()
        self.vector = y1[1:,]
     # the "update centroid" functions takes a datframe input of "clusters_vectors" and returns the mean value.
    def update_centroid(self, clusters_vectors):
        df = clusters_vectors
        self.vector = df.mean(axis=1)
# the cluster class below stores all the instances of the "objects"(data points)that have the same nearest centroid.
# in a list. this nearest centroid is also contained within the cluster object as "self.centroid_object".
class cluster:
    numb = 1
    def __init__(self,Centroid):
        self.label = cluster.numb
        cluster.numb +=1
        self.centroid_object = Centroid
        self.centroid_label = self.centroid_object.label
        self.objects_in_clust =[]
        self.object_vectors=[]
        self.object_vectors_concated = pd.DataFrame()
        self.a_values_dict = {}
        self.b_values_dict = {}
        
def get_sum_square(self):
    sum_dist_Sq = 0
    for obj in self.objects_in_clust:
        sum_dist_Sq += (obj.distances[self.centroid_label])
    return sum_dist_Sq
def concatenate_object_vectors(self):
    for ob in self.objects_in_clust:
        self.object_vectors.append(ob.vector)
    self.object_vectors_concated= pd.concat(self.object_vectors, axis=1)
def get_a_values(self):
    length = (np.shape(self.object_vectors_concated)[1])
    if length == 1:
        self.a_values_dict[0]=0
    if length == 2:
        x = self.object_vectors[0]
        y = self.object_vectors[1]
        distance = (np.linalg.norm(x-y))
        self.a_values_dict[0]=distance
        self.a_values_dict[1]=distance
    if length >2:
        for obj in range(0, length):
            temp_obj_list = self.object_vectors.copy()
            x = self.object_vectors[obj]
            temp_obj_list.pop(obj)
            rest_clust_concat = pd.concat(temp_obj_list, axis=1)
            distances = []
            for column in rest_clust_concat:
                y = rest_clust_concat[column]
                #should the distances below be squared?
                distances.append((np.linalg.norm(x-y)))
            temp_obj_list.append(obj)
            mean_distance = np.mean(distances)
            self.a_values_dict[obj] = mean_distance
def get_b_values(self, clusters):
    length = (np.shape(self.object_vectors_concated)[1])
    if length > 0:
    # possibility the nearest cluster is empty
        if self.object_vectors_concated.empty == True:
            pass
        else:
            other_cluster_distances = {}
            x = np.array(self.centroid_object.vector)
            for other_clust in clusters:
                if len(other_clust.objects_in_clust) > 0:
                    y = np.array(other_clust.centroid_object.vector)
                    euc_distance = (np.linalg.norm(x-y))
                    other_cluster_distances[other_clust.label] = euc_distance
            other_cluster_distances.pop(self.label)
            if len(other_cluster_distances) > 0:
                closest_clust_label = min(other_cluster_distances, key =other_cluster_distances.get)
                for clust in clusters:
                    if clust.label == closest_clust_label:
                        closest_clust = clust
                length_x = np.shape(self.object_vectors_concated)[1]
                length_y = np.shape(closest_clust.object_vectors_concated)[1]
                for obj in range(0,length_x):
                    x = np.array(self.object_vectors[obj])
                    distances = []
                    for column in range(0,length_y):
                        y_df = closest_clust.object_vectors_concated.iloc[:,column]
                        y = np.array(y_df)
                        distances.append((np.linalg.norm(x-y)))
                    mean_distance = np.mean(distances)
                    self.b_values_dict[obj] = mean_distance
                    
            if len(other_cluster_distances) == 0:
                length_x = np.shape(self.object_vectors_concated)[1]
                for obj in range(0,length_x):
                    self.b_values_dict[obj] = 0
                    
def get_silh_coef_for_objs(self):
    if len(self.objects_in_clust) >1:
        for obj_n in range(len(self.objects_in_clust)):
            coef = (self.b_values_dict[obj_n] - self.a_values_dict[obj_n]) / max(self.a_values_dict[obj_n], self.b_values_dict[obj_n])
            Object =self.objects_in_clust[obj_n]
            Object.silh_co = coef
    if len(self.objects_in_clust) == 1:
        for obj_n in range(len(self.objects_in_clust)):
            Object =self.objects_in_clust[obj_n]
            Object.silh_co = 0
# the "set_objects" function below takes in the dataframe of the dataset, and creates for each datapoint an instance
# of the "object"  class and stores them in the "object_list" list
def Set_objects(df, df_length, objects_list):
    for i in range(df_length):
        o = object(label = df.iloc[i,0], vector = df.iloc[i,1:])
        objects_list.append(o)
# the "make clusters" function takes a list containing all the objects, and a list of all the centroids.
# for each centroid, an empty cluster is initiated and saved to the clusters list. then, for each object,
# it groups ii with other objects that have the same nearest centroid.

def make_clusters(centroid_list,objects_list):
    # filling empty "clusters" dictionary  per centroid label, with empty list as key
    clusters_list = []
    for centroid in centroid_list:
        cluster_i = cluster(Centroid=centroid)
        clusters_list.append(cluster_i)
    # add each object to the list corresponding to its nearest neigbout
    # print("macro cluster target objects", len(objects_list))
    for objec in objects_list:
        for cluster_i in clusters_list:
            if cluster_i.centroid_label == objec.nearest_k:
                cluster_i.objects_in_clust.append(objec)
    return clusters_list
# the k_means function below creates the number of centroids, updating the vector attribute of the centroid instance
# using the "random_initliase function " that creates the vectors of an artificial datapoint within the dataset, not
# an actual datapoint.
# the " target clust" when K-means is used independently, and not within another alogrirhm, is simply the wholedataset
# the new leafs are the sub clusters formed from the target clust. See "make_clusters" functions comment for details.
# if the cluster isnt empty, the vectors of all the objects in the clusters are concatenated into one dataframe to allow
# the mean to be calculated for the purposes of updating centroid values.
def k_means(df, target_clust, K,):
    centroid_list = []
    max_iter = 10
    for k in range(1, K+1):
        centroid_object = centroid(k,0)
        centroid_object.random_initialise(df)
        centroid_list.append(centroid_object)
    for k in range(max_iter):
        for j in target_clust.objects_in_clust:
            j.get_distances(centroid_list)
            j.get_nearest_k()
        new_leafs = make_clusters(centroid_list, target_clust.objects_in_clust)
        for leaf in new_leafs:
            if len(leaf.objects_in_clust)>0:
                leaf.concatenate_object_vectors()
                clust_centroid = (leaf.centroid_object)
                clust_centroid.update_centroid(leaf.object_vectors_concated)
    return new_leafs
# the function below does preliminary formating before the main, "K-means" function is called.
# it takes the dataframe, creates an "object" class instance for every data point, intilises a
# null_centroid, for the purpose of creating a cluster that contains the whole dataset. This forms the
# target clust to be passed to the main "kmean" funcion.

def K_means_program(df, K):
    df_length = (np.shape(df)[0])-1
    objects_list = []
    Set_objects(df, df_length, objects_list)
    centroid_list_temp = []
    centroid_null = centroid(0, 0)
    centroid_list_temp.append(centroid_null)
    whole_dataset_cluster = make_clusters(centroid_list_temp, objects_list)
    return k_means(df, whole_dataset_cluster[0], K)
def largest_sum_square_clust(clusters_list):
    SS_list = []
    for Clust in clusters_list:
        SS_list.append(Clust.get_sum_square())
    return SS_list.index((max(SS_list)))
def smallest_sum_square_clust(clusters_list):
    SS_list = []
    for Clust in clusters_list:
        SS_list.append(Clust.get_sum_square())
    return SS_list.index((min(SS_list)))
# the "generate p dis" creates a probabiltiy distribution that, for each object, takes the squared eculaidan distance to its nearest centroid and
# appends it to a sum. After this, for each object, the "calc_prob" method is called. see the details for this in the object class comments.
def generate_p_dis(object_list):
    sum_dist_Sq = 0
    for obj in object_list:
        key_clust_min_dist = min(obj.distances)
        sum_dist_Sq += (obj.distances[key_clust_min_dist])
    for k in object_list:
            k.calc_prob(sum_dist_Sq)
# For "K-means_plus_program below, the first centroid is intiliased with the vector of a datapoint within the dataset using the "random_initialise_within_D" class method. see
# centroid class for more details.
# for each centroid chosen according to the Kmeans++ algorithm, the probability distribution is recalculated with the new distance to nearest centroid
# that subsequently occurs. The next centrpoid is chosen using the numpy random choice function that takes a list of the objects(an ascending index)
# and their corresponding probabilities. The "k index" varaible therefore is the index of the chosen object to become the next centroid.
# the remainder is simply the rest of the Kmeans code
def Kmeans_plus_program(K, maxiter, df):
    df_length = (np.shape(df)[0])-1
    objects_list = []
    Set_objects(df, df_length, objects_list)
    centroid_list = []
    centroid_1 = centroid(label=1, vector=0)
    centroid_1.random_initialise_within_D(df)
    centroid_list.append(centroid_1)

    for k in range(2, K+1):
        for j in objects_list:
            j.get_distances(centroid_list)
        generate_p_dis(objects_list)
        range_objects = list(range(1, len(objects_list)+1))
        probs_objects = []
        for i in objects_list:
            probs_objects.append(i.prob)
        K_index = (np.random.choice(range_objects, p = probs_objects,))
        K_object = objects_list[K_index]
        K_vector  = K_object.vector
        centroid_k = centroid(label=k, vector = K_vector)
        centroid_list.append(centroid_k)
    for i in range(maxiter):
        for i in objects_list:
            i.get_distances(centroid_list)
            i.get_nearest_k()
        clusters = make_clusters(centroid_list,objects_list)
        for clust in clusters:
            if (len(clust.objects_in_clust)) > 0:
                clust.concatenate_object_vectors()
                clust_centroid = (clust.centroid_object)
                clust_centroid.update_centroid(clust.object_vectors_concated)
    return clusters
# for the "bi_sect_Kmeans" function below, a dictionary is created with keys equal to the number of layers the devisive
 # heirarhcical tree will tak. Natrually the cluster with the largest sum of quared within each layer will be split using the
 # k-means algorithm. As such, the "tree list" variable is a list of all the non- split clusters which forms the final heirarhcical structure.
 # the first if statement simply bypasses the remainder of the code since the resultant clusture is the same as the wholedataset cluster
 # Otherwise, the code looks at the previous layer leafs stored int the H_tree_dictionary. and saves the cluster with the highest sum squared distancs
 # to the "target_cluster" variable that is then passed into the Kmeans algorithm (natrually with k=2) whilst any remaining clusters which have not been
 # split are saved to the final "Tree_list" list.
 
def Bisect_Kmeans(s, df):
    df_length = (np.shape(df)[0])-1
    # create dictioanry of decisive heirarchical tree
    n_tree_layers = s
    H_tree_dict = {}
    for i in range(1, n_tree_layers+1):
        H_tree_dict[i]=[]
    tree_list = []
    objects_list = []
    Set_objects(df, df_length, objects_list)

    centroid_list = []
    centroid_1 = centroid(label=1, vector=0)
    centroid_1.random_initialise(df)
    centroid_list.append(centroid_1)
    for j in objects_list:
        j.get_distances(centroid_list)
        j.get_nearest_k()
    clusters_i = make_clusters(centroid_list,objects_list)
    if s == 1:
        cluster_1 = clusters_i[0]
        cluster_1.concatenate_object_vectors()
        tree_list.append(clusters_i[0])
    if s != 1:
        maxSS_index_pos = largest_sum_square_clust(clusters_i)
        target_clust = clusters_i[maxSS_index_pos]
        H_tree_dict[1]= [target_clust]
        layers_remaining = (list(H_tree_dict.keys()))[1:]
        for layer in layers_remaining:
            prev_layer_leafs = H_tree_dict[layer-1]
            maxSS_index_pos = largest_sum_square_clust(prev_layer_leafs)
            target_clust = prev_layer_leafs[maxSS_index_pos]
            H_tree_dict[layer] = k_means(df, target_clust, 2)
            if len(prev_layer_leafs)>=2:
                minSS_index_pos = smallest_sum_square_clust(prev_layer_leafs)
                non_target_clust = prev_layer_leafs[minSS_index_pos]
                tree_list.append(non_target_clust)
            if layer == layers_remaining[-1]:
                for clus in H_tree_dict[layer]:
                    tree_list.append(clus)
    return tree_list
def get_mean_silhouette_coef(clusters):
    per_obj_coeff_list = []
    for cluster in clusters:
        for obj in cluster.objects_in_clust:
            per_obj_coeff_list.append(obj.silh_co)
    return np.mean(per_obj_coeff_list)
def dataset_silhouette_coef_for_clusters(clusters):
    for clust in clusters:
        clust.get_a_values()
        clust.get_b_values(clusters)
        clust.get_silh_coef_for_objs()
    return get_mean_silhouette_coef(clusters)
def print_cluster_contents(clusters):
    cluster_labels = {}
    for clust in clusters:
        cluster_labels[clust.label] =[]
        for obj in clust.objects_in_clust:
            cluster_labels[clust.label].append(obj.label)
    for key in cluster_labels.keys():
        print("objects in cluster:\n", cluster_labels[key])
        print("total number of objects in cluster above:", len(cluster_labels[key]))
        
def graph_maker(results_dictionary, parameter_k_bool , parameter_s_bool):
    if parameter_k_bool == True:
        K_i = list(results_dictionary.keys())
        Sil_co = list(results_dictionary.values())
        plt.plot(K_i , Sil_co)
        plt.xlabel('K value ')
        plt.ylabel('Silhouette coefficient')
        plt.show()
    if parameter_s_bool == True:
        K_i = list(results_dictionary.keys())
        Sil_co = list(results_dictionary.values())
        plt.plot(K_i , Sil_co)
        plt.xlabel('S value ')
        plt.ylabel('Silhouette coefficient')
        plt.show()
df = pd.read_table('dataset', delimiter = ' ', header = None)
random.seed(2)
np.random.seed(2)
print("kmeans full results\n")
K_means_results = {}
for i in range(1, 10):
    clusters_k_means = K_means_program(df, i)
    if i == 4:
        # print_cluster_contents(clusters_k_means)
        coeff_value = dataset_silhouette_coef_for_clusters(clusters_k_means)
        K_means_results[i] = coeff_value
    else:
        coeff_value = dataset_silhouette_coef_for_clusters(clusters_k_means)
        K_means_results[i] = coeff_value
for j in K_means_results.keys():
    print("K:", j, "Silhouette coef:", K_means_results[j])
print(graph_maker(K_means_results,parameter_k_bool= True, parameter_s_bool=False))
print("kmeans++ full results")
K_means_plus_results = {}
for i in range(1, 10):
    clusters_Kmeans_plus = Kmeans_plus_program(i, 5, df)
    if i == 4:
        # print_cluster_contents(clusters_Kmeans_plus)
        coeff_value = dataset_silhouette_coef_for_clusters(clusters_Kmeans_plus)
        K_means_plus_results[i] = coeff_value
else:

        coeff_value = dataset_silhouette_coef_for_clusters(clusters_Kmeans_plus)
        K_means_plus_results[i] = coeff_value
for j in K_means_plus_results.keys():
    print("K:", j, "Silhouette coef:", K_means_plus_results[j])
print(graph_maker(K_means_plus_results,parameter_k_bool= True,
parameter_s_bool=False))
print("bisecting full results")
bisec_K_means_results = {}
for i in range(1, 10):
    clusters_bisect = Bisect_Kmeans(i, df)
    if i == 4:
        # print_cluster_contents(clusters_bisect)
        coeff_value = dataset_silhouette_coef_for_clusters(clusters_bisect)
        bisec_K_means_results[i] = coeff_value
    else:
        coeff_value = dataset_silhouette_coef_for_clusters(clusters_bisect)
        bisec_K_means_results[i] = coeff_value
for j in bisec_K_means_results.keys():
    print("S:", j, "Silhouette coef:", bisec_K_means_results[j])
print(graph_maker(bisec_K_means_results,parameter_k_bool= False,
parameter_s_bool=True))
