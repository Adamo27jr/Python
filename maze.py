import graph
import render
import random
import coveringtree
import math
import copy

class Maze:
	def __init__(self, width, height, random=False, verticalSkewed=0, horizontalSkewed=0):
		# Defines the size of the maze
		self.width = width
		self.height = height

		# Parameter to make the generation of the maze random
		self.random = random

		# Allows to bias the generation of the maze in one direction or the other
		self.verticalSkewed = verticalSkewed
		self.horizontalSkewed = horizontalSkewed

		self.path = [self.initPos()]
		#self.path = [(5,4)]

		# Create the mesh of the maze
		self.graphe = self.generateGrid(self.width, self.height, self.random)
	
	def initPos(self):
		'''
		point aléatoirement placé dans le 1er quart du labyrinthe en bas à gauche
		'''
		posX = random.randint(0, math.floor(self.width/2)-1)
		posY = random.randint(0, math.floor(self.height/2)-1)
		return (posX, posY)
    


	def generateGrid(self, width, height, isRandom=False):
		"""
		generates and returns a non-oriented graph 
		"""
		graphe = graph.Graph()

		weight = 1

		# for each node create an arc to the right and another to the top except for the edges to create a mesh
		for x in range(width):
			for y in range(height):				
				if x<width-1:
					if isRandom:
						weight = random.uniform(0,1)
					graphe.add_arc_undirected( ((x,y),(x+1,y)), weight + self.verticalSkewed)
				if y<height-1:
					if isRandom:
						weight = random.uniform(0,1)
					graphe.add_arc_undirected( ((x,y),(x,y+1)), weight  + self.horizontalSkewed)
		return graphe
    
maze = Maze(10,10)
maze.display()

# 	def generateMaze(self):
# 		prim = coveringtree.algorithmePrim(self.graphe)
# 		graphe = graph.Graph()

# 		for node1, node2 in prim.items():
# 			if node2 is not None:
# 				# Method add_arc_undirected created two arc to make it undirected
# 				graphe.add_arc_undirected( (node1, node2) )
# 		
# 		self.graphe = graphe
# 	
# 	def display(self, draw_coordinates=False):
# 		"""
# 		displays the labyrinth graph on the screen
# 		draw_coordinates : mettre à True pour afficher les coordonnées des sommets
# 		"""
# 		render.draw_square_maze(self.graphe, self.path, draw_coordinates)
# 		
# 	def solve(self):
# 		'''
# 		Simple call to a method. 
# 		Structure set up in a future case of testing several resolution methods
# 		'''
# 		self.close_dead_end()
# 		
# 			
# 	def close_dead_end(self):
# 		grapheNoDeadEnd = copy.deepcopy(self.graphe)
# 		solution = [self.path[0]]

# 		nearNodes = [ [[0,0],[1,0]],
# 							[[1,0],[1,1]],
# 							[[1,1],[0,1]],
# 							[[0,1],[0,0]],]
# 		directionNode = [ (0,-1), (1,0), (0,1), (-1,0)]

# 		findDeadEnd = True
# 		while findDeadEnd :
# 			findDeadEnd = False

# 			for node in grapheNoDeadEnd.nodes():
# 				borderCounter = 0
# 				if node != solution[0] :#and node[0] !=0 and node[0] != self.width-1  and node[1] !=0 and node[1] != self.height-1:
# 					
# 					for testNode in nearNodes:
# 						node1 = (node[0]+testNode[0][0], node[1]+testNode[0][1])
# 						node2 = (node[0]+testNode[1][0], node[1]+testNode[1][1])

# 						if grapheNoDeadEnd.is_arc(node1, node2):
# 							borderCounter += 1

# 					if borderCounter == 3 :
# 						findDeadEnd = True
# 						for testNode in nearNodes:
# 							node1 = (node[0]+testNode[0][0], node[1]+testNode[0][1])
# 							node2 = (node[0]+testNode[1][0], node[1]+testNode[1][1])
# 							
# 							grapheNoDeadEnd.add_arc_undirected( (node1, node2) )
# 		#self.graphe = grapheNoDeadEnd
# 		
# 		#while solution[-1][0] !=0 and solution[-1][0] != self.width-1  and solution[-1][1] !=0 and solution[-1][1] != self.height-1:
# 		tempLen = 0
# 		while tempLen != len(solution) and solution[-1][0] < self.width-1  and solution[-1][1] < self.height-1:

# 			pos = solution[-1]
# 			tempLen = len(solution)
# 					
# 			secondPosChoice = None
# 			secondPosChoiceNextPosX = None
# 			secondPosChoiceNextPosY = None
# 			for next in range(len(directionNode)):
# 				testNode = nearNodes[next]
# 				node1 = (pos[0]+testNode[0][0], pos[1]+testNode[0][1])
# 				node2 = (pos[0]+testNode[1][0], pos[1]+testNode[1][1])

# 				if grapheNoDeadEnd.is_arc(node1, node2) == False:
# 					nextPosX = pos[0]+directionNode[next][0]
# 					nextPosY = pos[1]+directionNode[next][1]
# 					solution.append( (nextPosX, nextPosY) )

# 					# add an arc with weight of 0.5 to defined as already passed
# 					grapheNoDeadEnd.add_arc_undirected( (node1, node2), weight=0.5 )
# 				else:
# 					if grapheNoDeadEnd.arc_weight( (node1, node2)) == 0.5:
# 						nextPosX = pos[0]+directionNode[next][0]
# 						nextPosY = pos[1]+directionNode[next][1]
# 						secondPosChoiceNextPosX = node1
# 						secondPosChoiceNextPosY = node2
# 						secondPosChoice = (nextPosX, nextPosY)

# 			if tempLen == len(solution):
# 				grapheNoDeadEnd.set_arc_weight( (secondPosChoiceNextPosX, secondPosChoiceNextPosY), 1)
# 				grapheNoDeadEnd.set_arc_weight( (secondPosChoiceNextPosY, secondPosChoiceNextPosX), 1)
# 				if secondPosChoice == None:
# 					break
# 				else:
# 					solution.append( secondPosChoice )
# 						
# 		self.path = solution

# if __name__ == "__main__":
    
#     maze = Maze(10,10)
#     maze.display()
