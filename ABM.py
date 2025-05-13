import StA as StA

import math

import networkx as nx

import numpy as np

import random

import statistics as st

import seaborn as sns

import matplotlib.pylab as plt



def update_network(G, b=12, iteration=10, p=100, k = 2, isHomophily = False, budget = False, size =False,score_change = False, homophilyPercent =0, score_update_p = 1):

    '''

    :param G: network

    :param b: budget of agents

    :param d: each agent can send request to agents d hubs  away

    :ret

    '''

    Collab_networks = [G.copy()]

    RA_networks=[]

    att_dic= nx.get_node_attributes(G,'att')

    collab_network = G.copy()

    edge_list = [(u, v, 1) for (u, v) in G.edges()]

    collab_network.remove_edges_from(list(collab_network.edges()))

    collab_network.add_weighted_edges_from(edge_list)

    for i in range(iteration):

        ra_network = generate_ra_network(collab_network, b, k, p=p, isHomophily = isHomophily, budget = budget, homophilyPercent = homophilyPercent)

        nx.set_node_attributes(ra_network, att_dic, 'att')

        RA_networks.append(ra_network.copy())

        if score_change and i >1:

            update_network_score(collab_network, score_update_p)

        collab_network = update_collaboration_network(collab_network, ra_network)

        Collab_networks.append(collab_network.copy())

        if size:

            update_network_size(collab_network)

    return Collab_networks, RA_networks







def generate_ra_network(collab_network, b, d, p, isHomophily=False, budget =False, homophilyPercent= 0):

    node_list = list(collab_network.nodes())

    new_node_list = list(random.sample(node_list, len(node_list)))

    index_dic = dict(zip(node_list, range(len(node_list))))

    node_list = new_node_list



    edge_list = dict(zip((collab_network.edges()), [1] * len(collab_network.edges())))

    ra_network = nx.Graph()

    ra_network.add_nodes_from(node_list)

    agent_budget_dic = dict(zip(collab_network.nodes(), [b] * len(collab_network)))

    if budget:

        agent_budget_dic = dict(zip(collab_network.nodes(), [random.randint(6,18) for i in range(len(collab_network.nodes()))]))



    M = nx.adjacency_matrix(collab_network).toarray()

    new_M = compute_M(M, node_list, index_dic)

    M = new_M





    if d ==-1:

        AM1 = np.ones((len(M),len(M)))

    else:

        AM1 = AM_walk(M, k=d, p=1)



    success_dic = nx.get_node_attributes(collab_network, 'att')

    success_values = list(success_dic.values())

    success_values = [float(i) / max(success_values) for i in success_values]

    success_dic = dict(zip(success_dic.keys(), success_values))



    node_dic = dict(zip(node_list, range(len(collab_network))))

    ###

    matched_list = [(0, 0)]

    while len(agent_budget_dic) > 0 and len(matched_list) > 0:

        preference_array = generate_preference_list(AM1, node_dic, success_dic,

                                                    agent_budget_dic,p,ra_network, isHomophily=isHomophily, homophilyPercent= homophilyPercent)

        matched_list = MaximumStableMatching(node_list, preference_array)

        update_ra_network(ra_network, matched_list, edge_list, agent_budget_dic, list(collab_network.edges()))

    return ra_network





def MaximumStableMatching( node_list,preference_array):

    """

    :param node_list: node list to be matched

    :param preference_array: preference lists correponding to nodes

    :return: a list of matched pairs

    """

    #phase1:

    current_nodes = node_list.copy()

    current_preference =Symmetrize({current_nodes[i]:preference_array[i] for i in range(len(current_nodes))})

    matched =[]



    while len(current_nodes)>1:



        # proposal

        send_proposal = {}  # index in the preference list

        receive_proposal = {}  # index in the preference list

        while current_nodes:

            u = current_nodes.pop()

            i = 0

            if u in send_proposal:

                i = send_proposal[u] + 1

            while i < len(current_preference[u]):

                v = current_preference[u][i]

                if v not in receive_proposal:

                    send_proposal[u] = i

                    receive_proposal[v] = current_preference[v].index(u)

                    break

                else:

                    j = receive_proposal[v]

                    k = current_preference[v].index(u)

                    if k > j:

                        i += 1

                    else:

                        current_nodes.append(current_preference[v][j])

                        send_proposal[u] = i

                        receive_proposal[v] = k

                        break

        current_nodes = node_list.copy()







        # reduce lists

        new_matched, current_nodes, current_preference = ReducePreferenceLists(current_nodes, current_preference,

                                                                           send_proposal, receive_proposal)

        matched.extend(new_matched)

        # Phase 2

        while current_nodes:

            u = current_nodes.pop()

            if len(current_preference[u]) > 0:

                current_preference = FindRotation(current_preference, u)

                new_matched, current_preference, current_nodes = CheckMatchedNodes(current_preference)

                matched.extend(new_matched)

        #post process

        new_matched, current_preference, current_nodes = CheckRemainingNodes(matched, node_list,preference_array)

        matched.extend(new_matched)



    return matched





def CheckRemainingNodes(matched, node_list,preference_array):

    matched_nodes=[]

    for (u,v) in matched:

        matched_nodes.append(u)

        matched_nodes.append(v)

    unmatched = list(set(node_list)- set(matched_nodes))

    preference_dic={}

    for u in unmatched:

        l = preference_array[node_list.index(u)]

        preference_dic[u] = [x for x in l if x  in unmatched]

    preference_dic = Symmetrize(preference_dic)

    return CheckMatchedNodes(preference_dic)







def CheckMatchedNodes(current_preference):

    matched = []

    flag = True

    while flag:

        flag = False

        for u, l in current_preference.items():

            if len(l) == 1:

                matched.append((u, l[0]))

                matched.append((l[0],u))

                current_preference[u] =[]

                current_preference[l[0]] =[]

                flag = True

                current_preference = Symmetrize(current_preference)

    current_nodes = []

    for u, l in current_preference.items():

        if len(l) >0:

            current_nodes.append(u)

    return matched, current_preference,current_nodes



def ReducePreferenceLists(current_nodes, current_preference,send_proposal, receive_proposal):

    matched = []

    for u, l in current_preference.items():

        if u in receive_proposal:

            l_ = l[send_proposal[u]:receive_proposal[u] + 1]

            current_preference[u] = l_

            if len(l_) == 1:

                matched.append((u, l_[0]))

                #current_nodes.remove(u)

                current_preference[u] = []

        else:

            #current_nodes.remove(u)

            current_preference[u] = []

    current_preference = Symmetrize(current_preference)

    current_nodes = []

    for u, l in current_preference.items():

        if len(l) > 0:

            current_nodes.append(u)



    return matched, current_nodes, current_preference





def FindRotation(current_preference, u):

    As=[u]

    Bs=[]

    odd_rotation = False

    flag = True

    while flag:

        v = current_preference[u][1]

        Bs.append(v)

        w = current_preference[v][-1]

        if w not in As:

            As.append(w)

            u = w

        else:

            i = As.index(w)

            As = As[i:]

            Bs = Bs[i:]

            flag = False

    if set(As) ==set(Bs):

        odd_rotation = True



    if odd_rotation:

        current_preference[Bs[-1]] =[]

    else:

        for i in range(len(Bs)):

            try:

                l = current_preference[Bs[i]]

                current_preference[Bs[i]] = l[:l.index(As[i]) + 1]

                l = current_preference[As[i]]

                current_preference[As[i]] = l[l.index(Bs[i]):]

            except:

                aaa=0

    current_preference = Symmetrize(current_preference)

    return current_preference







def Symmetrize(preference_dic):

    """

    make sure that u is on v's list if and only if v is on  u's

    :param preference_dic:

    :return:

    """

    for u, l in preference_dic.items():

        rem =[]

        for v in l:

            if u not in preference_dic[v]:

                rem.append(v)

        preference_dic[u] = [x for x in l if x not in rem]

    return preference_dic



def update_network_score(collab_network, p):

    nodes = list(collab_network.nodes())

    scores = nx.get_node_attributes(collab_network, 'att')

    new_scores = {}

    for u in nodes:

        s1 = scores[u]

        s2 = 0

        s3 = 0

        s4 = []

        neighs = list(collab_network[u].keys())

        for v in neighs:

            w = collab_network[u][v]["weight"]

            s2 +=scores[v]*w

            s4.append(scores[v])

            s3 += w

        if len(s4)>0:

            s4.sort(reverse=True)

            x = (s4[0]+s4[1])/2

        if s3>0:

           # new_scores[u] = math.ceil(s1*p + (1-p)*(s2/s3))

            new_scores[u] = math.ceil(s1*p + (1-p)*(x))

        else:

            new_scores[u] = math.ceil(s1*p)

    nx.set_node_attributes(collab_network, new_scores, 'att')





def update_collaboration_network(collab_network, ra_network, t=0.1):

    edge_dic = {(u, v): (ra_network.get_edge_data(u, v)['weight']) / 4.0 for (u, v) in list(ra_network.edges())}



    edge_collab_list = {(u, v, collab_network.get_edge_data(u, v)['weight']) for (u, v) in list(collab_network.edges())}



    for (u, v, w) in edge_collab_list:

        if (u, v) in edge_dic:

            edge_dic[(u, v)] += w / 2

        elif (v, u) in edge_dic:

            edge_dic[(v, u)] += w / 2

        elif w / 2 >= t:

            edge_dic[(u, v)] = w / 2



    edge_list = [(u, v, round(w, 1)) for (u, v), w in edge_dic.items()]



    edges = list(collab_network.edges())

    collab_network.remove_edges_from(edges)

    collab_network.add_weighted_edges_from(edge_list)



    return collab_network





def update_network_size(collab_network):

    nodes = list(collab_network.nodes())

    n = int(len(nodes)/20)

    collab_network.remove_nodes_from(random.sample(nodes,n))

    a = max(nodes)

    new_nodes = range(a , a+2*n)

    selected = random.sample(list(collab_network.nodes()),2*n)

    edges = []

    for i in range(2*n):

        edges.append((new_nodes[i], selected[i], 1))

    collab_network.add_weighted_edges_from(edges)

    att = nx.get_node_attributes(collab_network, "att")

    group = nx.get_node_attributes(collab_network, "groups")



    a = int(max(att)/2)

    i = 0

    for u in new_nodes:

        i+=1

        if i %2 ==0:

            att[u] = random.randint(0,a)

        else:

            att[u] = 0

        group[u] = getGroups(att[u])

    nx.set_node_attributes(collab_network, att, "att")

    nx.set_node_attributes(collab_network, group, "groups")



def compute_M(M, nodes, index_dic):

    new_M = M.copy()

    i = -1

    for u in nodes:

        i += 1

        ind_u = index_dic[u]

        j = -1

        for v in nodes:

            j += 1

            ind_v = index_dic[v]

            val = M[ind_u][ind_v]

            new_M[i, j] = val

    return new_M







def AM_walk(M, k, p=-1):

    if k == -1:

        A = np.ones((len(M),len(M)))

        return A



    if p != -1:

        PM = p * np.array(M)

        A= PM.copy()

        current =  PM.copy()

        for i in range(k - 1):

            current = np.matmul(PM, current)

            A += current

    else:

        PM = np.array(M)

        A = (1/2)*PM.copy()

        current = PM.copy()

        for i in range(k - 1):

            current = np.matmul(PM, current)

            A += (1/(i+3))*np.array(current)

    return A









def generate_preference_list_(AM1, node_dic, success_dic, agent_budget_dic,p, ra_network,threshold=3, isHomophily=False):

    node_reversed_dic = dict(zip(node_dic.values(), node_dic.keys()))

    output_array = []

    ra_edge = {(u, v): ra_network.get_edge_data(u, v)['weight'] for (u, v) in list(ra_network.edges())}

    for node, i in node_dic.items():

        if node in agent_budget_dic:

            scores={}

            for j in range(len(AM1)):

                item = node_reversed_dic[j]

                if AM1[i, j] > 0 and node != item:

                    #if (node, item) not in edge_list and (item, node) not in edge_list:

                    if agent_budget_dic[node] > 0:

                        if ra_edge.get((node, item), 0) + ra_edge.get((item, node), 0) <threshold:

                            if isHomophily == False:

                                x= random.uniform(0, 1)

                                if x<=p:

                                    scores[item] =success_dic[item]

                                else:

                                    scores[item] = random.uniform(0, 1)

                            else:

                                x = random.uniform(0, 1)

                                if x <= p:

                                    scores[item] = abs(success_dic[item] - success_dic[node])

                                else:

                                    scores[item] = random.uniform(0, 1)



            keys = list(scores.keys())

            random.shuffle(keys)

            scores2 = {k: scores[k] for k in keys}

            scores= scores2



            if isHomophily == False:

                prference_list = [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

            else:

                prference_list = [k for k, v in sorted(scores.items(), key=lambda item: item[1])]



            output_array.append(prference_list)

        else:

            output_array.append([])



    return output_array







def update_ra_network(ra_network, matched_list, edge_list, agent_budget_dic, collab_edges):

    # a = list(ra_network.get_edge_data())

    edge_dic = {(u, v): ra_network.get_edge_data(u, v)['weight'] for (u, v) in list(ra_network.edges())}



    for (u, v) in matched_list:

        w = 1 if (u, v) in collab_edges or (v, u) in collab_edges or (u, v) in edge_dic or (v, u) in edge_dic  else 1

        if (u, v) in edge_dic:

            edge_dic[(u, v)] = edge_dic[(u, v)] + w

        elif (v, u) in edge_dic:

            edge_dic[(v, u)] = edge_dic[(v, u)] + w

        else:

            edge_dic[(u, v)] = w

        if u in agent_budget_dic:

            agent_budget_dic[u] -= w

            if agent_budget_dic[u] == 0:

                del agent_budget_dic[u]

        if v in agent_budget_dic:

            agent_budget_dic[v] -= w

            if agent_budget_dic[v] == 0:

                del agent_budget_dic[v]







    edge_list2 = [(u, v, w) for (u, v), w in edge_dic.items()]

    edge_list.update(edge_dic)

    ra_network.add_weighted_edges_from(edge_list2)



    # return ra_network









def generate_preference_list(AM1, node_dic, success_dic, agent_budget_dic,p, ra_network,threshold=3, isHomophily=False, homophilyPercent=0):

    node_reversed_dic = dict(zip(node_dic.values(), node_dic.keys()))

    output_array = []

    ra_edge = {(u, v): ra_network.get_edge_data(u, v)['weight'] for (u, v) in list(ra_network.edges())}

    for node, i in node_dic.items():

        if node in agent_budget_dic:

            scores_hom={}

            scores_sc={}

            for j in range(len(AM1)):

                item = node_reversed_dic[j]

                if AM1[i, j] > 0 and node != item:

                    if agent_budget_dic[node] > 0:

                        if ra_edge.get((node, item), 0) + ra_edge.get((item, node), 0) <threshold:

                            scores_sc[item] = success_dic[item]

                            scores_hom[item] = abs(success_dic[item] - success_dic[node])



            keys = list(scores_hom.keys())

            random.shuffle(keys)

            scores_hom = {k: scores_hom[k] for k in keys}

            scores_hom_ = [k for k, v in sorted(scores_hom.items(), key=lambda item: item[1], reverse=True)]

            scores_sc = {k: scores_sc[k] for k in keys}

            scores_sc_ = [k for k, v in sorted(scores_sc.items(), key=lambda item: item[1])]

            prference_list =[]

            while len(keys)>0:

                x = random.uniform(0, 1)

                if x < p:

                    item = keys.pop()

                    prference_list.append(item)

                    scores_hom_.remove(item)

                    scores_sc_.remove(item)

                else:

                    x = random.uniform(0, 1)

                    if x >= homophilyPercent:

                        item = scores_sc_.pop()

                        prference_list.append(item)

                        scores_hom_.remove(item)

                        keys.remove(item)

                    else:

                        item = scores_hom_.pop()

                        prference_list.append(item)

                        scores_sc_.remove(item)

                        keys.remove(item)

            output_array.append(prference_list)

        else:

            output_array.append([])



    return output_array





def getGroups(i):

    if i <3:

        return 0

    elif i <7:

        return 1

    elif i <15:

        return 2

    else:

        return 3









def example():

    Graphs = gen_graphs(i=5, score_dist=True)



    models = ["Complete", "Tree", "Erdos-Renyi"]





    modelC=-1

    for model in Graphs:

        print("########################")

        modelC+=1

        print(models[modelC])

        matrix_ = np.zeros(shape=(8, 5))

        probabilities = [0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1]

        probabilities2 = [0, 0.25, 0.5, 0.75, 1]



        for i in range(len(probabilities)):

            for j in range(len(probabilities2)):

                r=[]

                for g in model:

                    r.append(model_simulation_1(g.copy(), percent = probabilities[i], iteration=15, k=-1, homophilyPercent = probabilities2[j]))

                matrix_[i][j] = st.mean(r)

                print("$$$$$$$$$$")

                print(probabilities[i])

                print(probabilities[j])

                print(matrix_[i][j])



        plot_dic(matrix_, models[modelC] + "_1_t1",  sym =False)



def model_simulation_1(G, percent, iteration=20, k = 2, isHomophily = False, budget = False, size = False, score_change = False, homophilyPercent = 0.0):

    collab_networks, ra_networks = update_network(G, p= percent , iteration= iteration, k=k, isHomophily = isHomophily, budget=budget, size = size, score_change = score_change, homophilyPercent = homophilyPercent)

    return StA.maxStrat(collab_networks[-1], 4)



def plot_dic(input, name,  sym =False):

    sns.set(font_scale=2)

    if sym:

        mask = np.zeros_like(input)

        mask[np.triu_indices_from(mask)] = True

        with sns.axes_style("white"):

            ax = sns.heatmap(input, cmap="Blues", mask=mask,  square=True)

           # ax.set(xlabel ='Collaboration Score Class', ylabel = 'H-index Class')

            plt.savefig( name + '.png')

            plt.close()

    else:

        #ax = sns.heatmap(input, cmap="rocket_r", linewidth=0.5)

        ax = sns.heatmap(input, cmap="vlag", linewidth=0.5)

        #ax.set(xlabel='Collaboration Score Class', ylabel='H-index Class')

        plt.savefig( name + 'vlag.pdf')



        plt.close()



def gen_graphs(n = 50, i = 10,score_dist=False, k=1.9):

    if score_dist:

        k = -1

    g1 = gen_compgraphs_k(n, k=k, i = i)

    g1 = [set_groups(g) for g in g1]

    g2 = gen_trees_k(n, k=k, i = i)

    g2 = [set_groups(g) for g in g2]

    g3 = gen_erdosgraphs_k(n, k=k, i = i)

    g3 = [set_groups(g) for g in g3]



    Graphs =[g1, g2, g3]

    return Graphs





def gen_trees_k(n, k=2, i = 20):

    Gs=[]

    for _ in range(i):

        if k == -1:

            s = list(range(n))

        else:

            s = nx.utils.powerlaw_sequence(n, k)

            s = update_list(s)

        G = nx.random_tree(n)

        s = [round(item) for item in s]

        score_dic = dict(zip(G.nodes(), s))

        nx.set_node_attributes(G, score_dic, 'att')

        Gs.append(G.copy())



    return Gs





def gen_erdosgraphs_k(n, k=2, i = 20):

    Gs=[]

    for _ in range(i):

        if k == -1:

            s = list(range(n))

        else:

            s = nx.utils.powerlaw_sequence(n, k)

            s = update_list(s)

        G = nx.erdos_renyi_graph(n, 0.1)

        s = [round(item) for item in s]

        score_dic = dict(zip(G.nodes(), s))

        nx.set_node_attributes(G, score_dic, 'att')

        Gs.append(G.copy())



    return Gs





def set_groups(G):

    att_dic = nx.get_node_attributes(G, 'att')

    for u,v in att_dic.items():

        if v>100:

            att_dic[u]=100

    nx.set_node_attributes(G, att_dic, 'att')

    group_dic = {k:getGroups(v) for (k, v) in att_dic.items()}

    nx.set_node_attributes(G,group_dic, 'groups')

    return G



def gen_compgraphs_k(n, k=2, i = 5):

    Gs=[]

    for _ in range(i):

        if k ==-1:

            s = list(range(n))

        else:

            s = nx.utils.powerlaw_sequence(n, k)

            s = update_list(s)

        G = nx.complete_graph(n)

        s = [round(item) for item in s]

        score_dic = dict(zip(G.nodes(), s))

        nx.set_node_attributes(G, score_dic, 'att')

        Gs.append(G.copy())



    return Gs





def update_list(l):

    l = [ math.ceil(i)  for i in l]

    n = len(l)

    l.sort()

    map_ ={}

    counter= n+1

    for i in l:

        if i not in map_:

            if i <n:

                map_[i] = i

            else:

                map_[i] = counter

                counter+=1

    output =[map_[i] for i in l]

    return output





example()