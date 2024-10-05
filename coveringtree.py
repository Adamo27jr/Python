import copy

def algorithmePrim(graphe):
    # Implementation of Prim's algorithm on a graph

    inf = float ( " inf ")

    # makes it easier to read the position of nodes
    x = 0
    y = 1

    # used to know the edges of the labyrinth
    mazeHeight = max(graphe.nodes())[0]
    mazeWidth = max(graphe.nodes())[1]

    # dictionnaire qui pour chaque sommet v associera le coût de connexion de v à l'arbre
    connectionCost = {}

    # dictionnaire qui pour chaque sommet v associera le parent de v dans l'arbre généré
    parent = {}

    for v in graphe.nodes() :
        connectionCost[v] = inf
        parent[v] = None

    priorityGraphe = copy.deepcopy(graphe)

    while len(priorityGraphe.nodes())>0:
        # un sommet dans Q (priorityGraphe) tel que C[u] est minimal
        minimal = inf
        u = None
        for sommet in priorityGraphe.nodes():
            if u == None or connectionCost[sommet] < minimal:
                minimal = connectionCost[sommet]
                u = sommet
	
		# Q ← Q \ {u}
        priorityGraphe.remove_node(u)
				
        # pour tout v successeur de u dans G 
        for v in list(graphe.successors(u)):
            # if v est sur une des deux frontières verticales du labyrinthe then
            if v[x] == 0 or  v[x] == mazeWidth:
                # if u est situé juste en bas de v then
                if u[y]<v[y]:
                    connectionCost[v] = 0
                    parent[v] = u
            # if v est sur une des deux frontières horizontales du labyrinthe then
            if v[y] == 0 or  v[y] == mazeHeight :
                # if u est situé juste à gauche de v then
                if u[x]<v[x]:
                    connectionCost[v] = 0
                    parent[v] = u
            # if v ∈ Q and weight(u, v) < C[v] then
            if v in priorityGraphe.nodes() and  graphe.arc_weight( (u,v) ) < connectionCost[v]:
                connectionCost[v] = graphe.arc_weight( (u,v) )
                parent[v] = u
    return parent