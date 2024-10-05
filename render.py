import matplotlib.pyplot as plot
import math

"""
Ce module a été écrit par Pacôme Perrotin en octobre 2021 pour le projet Mazes.
Ce projet a ensuite été repris et modifié par Raphaël Gass en octobre 2022.

Ce module met à votre disposition des fonctions d'affichages de labyrinthes,
qui utilisent pyplot. 

Les fonctions qui vous seront utiles sont draw_tree, draw_square_maze et
draw_hex_maze. Les détails d'utilisation de chaque fonction sont à trouver
dans les commentaires juste avant chaque fonction.

(!!!) IMPORTANT : Toutes les fonctions de ce modules font l'hypothèse que
les sommets des graphes donnés en paramètre sont de la forme (x, y),
pour x et y deux entiers.
Si votre graphe utilise des sommets nommés différemment, l'affichage échouera
et produira probablement un crash.
"""





"""
Fonction draw_tree

paramètre graph : graphe à afficher, de la classe graph.Graph
paramètre optionnel draw_coordinates : à mettre à True pour afficher les coordonnées des sommets

Affiche le graphe donné en entrée de façon très simple. Permet de voir
la structure du graphe de façon nue.
Cette fonction ne renvoie rien.
"""

def draw_tree(graph, draw_coordinates=False):
  for u in graph.nodes():
    x, y = u
    
    if draw_coordinates:
      plot.text(x, y, str(x) + ', ' + str(y),
        verticalalignment = 'center',
        horizontalalignment = 'center')
        
    for v in graph.successors(u):
      draw_line(u, v)
      
  plot.axis('scaled')
  plot.show()



"""
Fonction draw_square_maze

paramètre graph : graphe à afficher, de la classe graph.Graph
paramètre optionnel path : une liste de coordonnées qui représentent un chemin
paramètre optionnel draw_coordinates : à mettre à True pour afficher les coordonnées des sommets

Affiche le labyrinthe passé en paramètre, dont l'arbre est encodé dans graph.
Si path est un chemin non vide, ce chemin sera dessiné en rouge. Cela vous
permet d'afficher la solution du labyrinthe une fois que vous l'avez
calculée.

La paramètre optionnel draw_coordinates, une fois mis à True, permet d'afficher
les coordonnées de chaque case du labyrinthe. Utile pour débugger.

IMPORTANT : le graphe passé en paramètre doit être non-orienté. C'est à dire
que pour toute arête (a, b), l'arête (b, a) doit également faire partie du
graphe. Dans le cas contraire, toute arête incomplète sera dessinée comme un
mur, comme s'il n'y avait pas d'arête.
"""

def draw_square_maze(graph, path = [], draw_coordinates=False):
  x0, y0 = path[0]  
  plot.scatter(x0+0.5,y0+0.5, color='r', s=100)
    
  for u in graph.nodes():
    x, y = u
    
    if draw_coordinates:
        plot.text(x, y, str(x) + ', ' + str(y),
        verticalalignment = 'center',
        horizontalalignment = 'center')
        
    for v in graph.successors(u):
        draw_line(u, v)

    for k in range(len(path) - 1):  
        x1, y1 = path[k]
        x2, y2 = path[k + 1]
        plot.gca().add_line( plot.Line2D( (x1+0.5, x2+0.5), (y1+0.5, y2+0.5), color='r' ) )
  

  plot.axis('scaled')
  plot.show()
  
  
  
"""
Fonction draw_hex_maze

paramètre graph : graphe à afficher, de la classe graph.Graph
paramètre optionnel path : une liste de coordonnées qui représentent un chemin
paramètre optionnel draw_coordinates : à mettre à True pour afficher les coordonnées des sommets

Affiche le labyrinthe hexagonal passé en paramètre,
dont l'arbre est encodé dans graph.
Si path est un chemin non vide, ce chemin sera dessiné en rouge. Cela vous
permet d'afficher la solution du labyrinthe une fois que vous l'avez
calculée.

La paramètre optionnel draw_coordinates, une fois mis à True, permet d'afficher
les coordonnées de chaque case du labyrinthe. Utile pour débugger, et pour
pouvoir avoir une bonne idée de comment une grille hexagonale est organisée.

IMPORTANT : le graphe passé en paramètre doit être non-orienté. C'est à dire
que pour toute arête (a, b), l'arête (b, a) doit également faire partie du
graphe. Dans le cas contraire, toute arête incomplète sera dessinée comme un
mur, comme s'il n'y avait pas d'arête.
"""

def draw_hex_maze(graph, path = [], draw_coordinates=False):

  for u in graph.nodes():
    x, y = u
    
    if draw_coordinates:
      xt, yt = graph_to_hex_grid(u)
      plot.text(xt, yt, str(x) + ', ' + str(y),
        verticalalignment = 'center',
        horizontalalignment = 'center')
    
    if y % 2 == 0:
      neighbours = [(x + 1, y), (x, y - 1), (x - 1, y - 1), (x - 1, y),
        (x - 1, y + 1), (x, y + 1)]
    else :
      neighbours = [(x + 1, y), (x + 1, y - 1), (x, y - 1), (x - 1, y),
        (x + 1, y + 1), (x, y + 1)]
    for v in [s for s in neighbours if s not in graph.successors(u)]:
      draw_hex_wall(u, v)

  for k in range(len(path) - 1):  
    x1, y1 = graph_to_hex_grid(path[k])
    x2, y2 = graph_to_hex_grid(path[k + 1])
    plot.gca().add_line( plot.Line2D( (x1, x2), (y1, y2), color='r' ) )
  
  plot.axis('scaled')
  plot.show()
  
  
  
  
  
  
  
  
  
  

"""
Ce qui suit sont les fonctions interne du module. Il n'est pas nécessaire de
les utiliser pour votre projet.
"""

def draw_line(u, v, color=None):
  xu, yu = u
  xv, yv = v
  plot.gca().add_line( plot.Line2D( (xu, xv), (yu, yv), color=color ) )



def draw_node_box(u, color=None):
  x, y = u
  draw_line( (x - .5, y - .5), (x + .5, y - .5), color )
  draw_line( (x - .5, y + .5), (x + .5, y + .5), color )
  draw_line( (x + .5, y - .5), (x + .5, y + .5), color )
  draw_line( (x - .5, y - .5), (x - .5, y + .5), color )

def draw_node_box_corners(u, color=None):
  x, y = u
  draw_line( (x - .5, y - .5), (x - .5, y - .5), color )
  draw_line( (x - .5, y + .5), (x - .5, y + .5), color )
  draw_line( (x + .5, y - .5), (x + .5, y - .5), color )
  draw_line( (x - .5, y - .5), (x - .5, y - .5), color )
  
def draw_square_wall(u, v, color=None):
  xu, yu = u
  xv, yv = v
  xm, ym = ( (xu + xv) / 2, (yu + yv) / 2 )
  
  a = (xm + (ym - yu), ym + (xm - xu))
  b = (xm - (ym - yu), ym - (xm - xu))
  
  draw_line(a, b, color)

# The hexagon flat to flat diameter is 1. As such the horizontal distance
# between 2 nodes is always 1.
# The vertical distance is the long flat side of a triangle of small side 0.5
# and hypothenus 1.0. v^2 + 0.5^2 = 1.0^2, so v = sqrt( 1 - 0.25 ) = sqrt( 3/4 ).
# Every odd layer is offset by 0.5.

horizontal_distance = 1
vertical_distance = math.sqrt( 3 / 4 )
horizontal_offset = .5
  
def graph_to_hex_grid(u):
  x, y = u
  return (x + (horizontal_offset if y % 2 == 1 else 0), y * vertical_distance)
  
# the ratio from the flat radius r to get the side t is 2 / sqrt(3)

radius_to_side_ratio = 2 / math.sqrt(3)
  
  
def rotate_vector(v, angle, scale = 1.):
  x, y = v
  return ( (x * math.cos(angle) - y * math.sin(angle)) * scale,
           (x * math.sin(angle) + y * math.cos(angle)) * scale)
  
def draw_hex_wall(u, v, color=None):
  xu, yu = graph_to_hex_grid(u)
  xv, yv = graph_to_hex_grid(v)
  
  xm, ym = ( (xu + xv) / 2, (yu + yv) / 2 )
  
  #     a
  # u   m   v
  #     b
  
  a = rotate_vector( (xv - xm, yv - ym), math.pi / 2, radius_to_side_ratio / 2)
  b = rotate_vector( (xu - xm, yu - ym), math.pi / 2, radius_to_side_ratio / 2)
  
  xa, ya = a
  xb, yb = b
  
  draw_line((xa + xm, ya + ym), (xb + xm, yb + ym), color)


