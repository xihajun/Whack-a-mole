import numpy as np
import tkinter as tk
import sys

size = input("size of the game:")
size = int(size)

class Game:

    def __init__(self, master, grid=np.diag(np.ones(size))):
        # print(type(master))
        frame = tk.Frame(master)
        frame.pack()
        self.grid = grid
        self.size = size
        self.counter = 0
        self.buttons = [[None for _ in range(size)] for _ in range(size)]
        self.map = np.random.randint(0,2,(size,size))
        
        # set button icon (mouse emoji is not supported for some reason)
        for i in range(size):
            for j in range(size):
                if self.map[i][j] == 0:
                    self.buttons[i][j] = tk.Button(frame, text="⭕️", 
                            command=lambda i=i, j=j: self.hit(i,j))
                    self.buttons[i][j].grid(row=i, column=j)
                else:
                    self.buttons[i][j] = tk.Button(frame, text="❌", 
                            command=lambda i=i, j=j: self.hit(i,j))
                    self.buttons[i][j].grid(row=i, column=j)
    
    def hit(self, row, col):
        '''Hit Action'''
        # out of box situation, doing nothing
        if row not in range(0,self.size):
            self.printemoji()
            return
        if col not in range(0,self.size):
            self.printemoji()
            return
        if self.map[row][col] != 1:
            self.printemoji()
            return
        else:
            # hit row,col
            self.rm(row,col)
            [self.add(i,j) for i,j in self.getneibours(row,col)]
            self.printemoji()
            return
        
    def getneibours(self, row, col):
        '''Return neibours list [[rows, cols]]'''
        # max: 4 neibours
        neibours = []
        for i,j in [[row-1,col],[row+1,col],[row,col-1],[row,col+1]]:
            if i>=0 and j>=0 and i<self.size and j<self.size:
                neibours.append([i,j])
        return neibours
            
    def rm(self,row,col):
        '''remove the critters'''
        self.map[row][col] = 0
        self.buttons[row][col]["text"] = "⭕️"
        
    def add(self,row,col):
        '''Up or DOWN'''
        self.map[row][col] += 1
        self.map[row][col] %= 2
        if self.map[row][col] == 0:
            self.buttons[row][col]["text"] = "⭕️"
        else:
            self.buttons[row][col]["text"] = "❌"

    # visualisation version-0
    def printGame(self):
        print(self.map)
    
    # visualisation version-0.5
    def printemoji(self):
        if self.counter > 0:
            for i in range(self.size):
                sys.stdout.write("\033[F")
        for row in self.map:
            for col in row:
                if col == 0:
                    print("⭕️",end=" ")
                else:
                    print("🐭",end=" ")
            print("")
        self.counter = self.counter + 1
# Run GUI
root = tk.Tk()
root.title("Whack a mole")
app = Game(root)
root.mainloop()
