from PIL import Image
import numpy as np
from collections import deque


def find_entrances(maze):
    (height, width) = maze.shape

    # top border
    ind = np.squeeze(np.argwhere(np.diff(maze[0,:]) != 0))
    top_entrances = [((0,ind[i]+1), (0,ind[i+1])) for i in range(0, len(ind), 2)]
    # bottom border
    ind = np.squeeze(np.argwhere(np.diff(maze[height-1,:]) != 0))
    bottom_entrances = [((height-1,ind[i]+1), (height-1,ind[i+1])) for i in range(0, len(ind), 2)]
    # left border
    ind = np.squeeze(np.argwhere(np.diff(maze[:,0]) != 0))
    left_entrances = [((ind[i]+1,0), (ind[i+1],0)) for i in range(0, len(ind), 2)]
    # right border
    ind = np.squeeze(np.argwhere(np.diff(maze[:,width-1]) != 0))
    right_entrances = [((ind[i]+1,width-1), (ind[i+1],width-1)) for i in range(0, len(ind), 2)]

    return top_entrances + bottom_entrances + left_entrances + right_entrances


def is_valid_move(maze, visited, row, col):
    return (0 <= row < maze.shape[0]) and (0 <= col < maze.shape[1]) and (maze[row,col] == 1) and (not visited[row,col])


def bfs(maze, start, possible_ends):
    height, width = maze.shape
    visited = np.zeros((height, width))
    queue = deque([(start, 0)])
    visited[start] = 1

    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) in possible_ends:
            return distance + 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if is_valid_move(maze, visited, new_row, new_col):
                visited[new_row][new_col] = 1
                queue.append(((new_row, new_col), distance + 1))
    return -1

def bfs_with_teleports(maze, start, possible_ends, teleports):
    height, width = maze.shape
    visited = np.zeros((height, width))
    queue = deque([(start, 0)])
    visited[start] = 1

    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) in possible_ends:
            return distance + 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if is_valid_move(maze, visited, new_row, new_col):
                visited[new_row][new_col] = 1
                queue.append(((new_row, new_col), distance + 1))
        if (row, col) in teleports:
            other_teleports = [teleport for teleport in teleports if teleport != (row, col)]
            for teleport in other_teleports:
                (tel_row, tel_col) = teleport
                if is_valid_move(maze, visited, tel_row, tel_col):
                    visited[tel_row][tel_col] = 1
                    queue.append(((tel_row, tel_col), distance + 1))

    return -1
 

def find_shortest_path(maze, entrances, teleports = []):

    def expand_entrance(entrance):
        if not np.abs(entrance[0][0]-entrance[1][0]):
            const, start_y = entrance[0]
            const, end_y = entrance[1]
            expanded_entrance = [(const, y) for y in range(start_y, end_y + 1)]
        else:
            start_x, const = entrance[0]
            end_x, const = entrance[1]
            expanded_entrance = [(x, const) for x in range(start_x, end_x + 1)]
        
        return expanded_entrance
    
    N = len(entrances)
    expanded_entrances = [expand_entrance(entrances[i]) for i in range(N)]

    shortest_path = float('inf')
    for i in range(N):

        entrance = expanded_entrances[i]
        possible_exits = []
        for j in range(N):
            if j != i:
                possible_exits += expanded_entrances[j]
        
        for start in entrance:
            if not teleports:            
                distance = bfs(maze, start, possible_exits)
            else:
                distance = bfs_with_teleports(maze, start, possible_exits, teleports)
            if (distance != -1) and distance < shortest_path:
                shortest_path = distance

    if shortest_path == float('inf'):
        shortest_path = -1

    return shortest_path

# ---main---

image_path = r'public3\set\09.png'
#image_path = input()
N_teleports = int(input())
teleports = [tuple(map(int, input().split())) for _ in range(N_teleports)]

image = Image.open(image_path)
image = image.convert("L")
maze = np.array(image)/np.max(image)

entrances = find_entrances(maze)
N_entrancies = len(entrances)

shortest_path = find_shortest_path(maze, entrances)

print(N_entrancies)
print(shortest_path)
if N_teleports <= 1:
    print(shortest_path)
else:
    shortest_path_with_teleports = find_shortest_path(maze, entrances, teleports)
    print(shortest_path_with_teleports)




