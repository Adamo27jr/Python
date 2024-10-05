import maze


'''
Instantiate the labyrinth 
Maze(width, height, random=False, verticalSkewed=0, horizontalSkewed=0)
width : width of the maze
height : height of the maze
random : if true generate a random maze
verticalSkewed : change the weight of the vertical branches 
horizontalSkewed : change the weight of the horizontal branches 
'''
maze = maze.Maze(10, 10, random=True, horizontalSkewed=0, verticalSkewed=0)


# generates the labyrinth (Pim's algorithm)
# maze.generateMaze()

# Display the maze
maze.display()

# Solve the maze
maze.solve()

# Display the maze solved
# maze.display()
