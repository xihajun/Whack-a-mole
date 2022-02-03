import numpy as np

class Game:
    def __init__(self, size, **kwargs):

        self.size = size
        maparg = kwargs.get('map', None)
        if maparg is None:
            self.map = np.random.randint(0,2,(size,size))
        else:
            if maparg.size == size**2:
                self.map = maparg
            else:
                print("Double Check the size or map")

    def printGame(self):
        print(self.map)

    def printemoji(self):
        for row in self.map:
            for col in row:
                if col == 0:
                    print("⭕️",end=" ")
                else:
                    print("🐭",end=" ")
            print("")

    def getneibours(self, row, col):
        # max: 4 neibours
        neibours = []
        for i,j in [[row-1,col],[row+1,col],[row,col-1],[row,col+1]]:
            if i>=0 and j>=0 and i<self.size and j<self.size:
                neibours.append([i,j])
        return neibours

    def rm(self,row,col):
        self.map[row][col] = 0

    def add(self,row,col):
        self.map[row][col] += 1
        self.map[row][col] %= 2

    def hit(self, row, col):
        row = row - 1
        col = col - 1
        if row not in range(0,self.size):
            self.printemoji()
            return
        if col not in range(0,self.size):
            self.printemoji()
            return
        if self.map[row][col] != 1:
            # nothing happens when hit the air
            self.printemoji()
            return
        else:
            # hit row,col
            self.rm(row,col)
            [self.add(i,j) for i,j in self.getneibours(row,col)]
            self.printemoji()
            return

    def play(self):
        self.printemoji()
        while sum(sum(self.map))>0 or inputs=='exit':
            inputs = input("Choose row and col (eg. 1 2) or exit:")
            if inputs == 'exit':
                break
            row, col = inputs.split(" ")
            self.hit(int(row),int(col))
        print("Done👍")

test = Game(size=2)
test.play()