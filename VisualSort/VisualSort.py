import time
import wave
import numpy as np
import scipy as sp
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class TrackedArray():

    def __init__(self, arr, kind="minimal"):
        self.arr = np.copy(arr)
        self.kind = kind
        self.reset()

    def reset(self):
        self.indices = []
        self.values = []
        self.access_type = []
        self.full_copies = []

    def track(self, key, access_type):
        self.indices.append(key)
        self.values.append(self.arr[key])
        self.access_type.append(access_type)
        if self.kind == "full":
            self.full_copies.append(np.copy(self.arr))

    def GetActivity(self, idx=None):
        if isinstance(idx, type(None)):
            return [(i, op) for (i, op) in zip(self.indices, self.access_type)]
        else:
            return (self.indices[idx], self.access_type[idx])

    def __delitem__(self, key):
        self.track(key, "del")
        self.arr.__delitem__(key)

    def __getitem__(self, key):
        self.track(key, "get")
        return self.arr.__getitem__(key)

    def __setitem__(self, key, value):
        self.arr.__setitem__(key, value)
        self.track(key, "set")

    def __len__(self):
        return self.arr.__len__()

    def __str__(self):
        return self.arr.__str__()

    def __repr__(self):
        return self.arr.__repr__()

plt.rcParams["figure.figsize"] = (12,8)
plt.rcParams["font.size"] = 16

#needs to add a more aesthetically pleasing timer
#shows in cmd window and only on bubble need to place it on others
#also adding it to the graph GUI
def tstart():
    t0 = time.perf_counter()
    return t0

def tstop(t0):
    return time.perf_counter() - t0

#this is used to display the different colored bars depending on get or set
def update(frame, nums, container):
    for rectangle, height in zip(container.patches, nums.full_copies[frame]):
        rectangle.set_height(height)
        rectangle.set_color("#1f77b4")
    idx, op = nums.GetActivity(frame)
    if op == "get":
        container.patches[idx].set_color("magenta")
    elif op == "set":
        container.patches[idx].set_color("red")
    return(*container,)

def bubble_sort(nums):
    t0 = tstart()
    n = len(nums)
    x = np.arange(0,n,1)
    for i in range(n):
        for j in range(0, n-i-1):
            if nums[j] > nums[j+1]:
                nums[j] , nums[j+1] = nums[j+1], nums[j]
    print(tstop(t0))


def insert_sort(nums):
    for x in range(1,len(nums)):
        key = nums[x]
        j = x - 1
        while j >= 0 and key < nums[j]:
            nums[j+1] = nums[j]
            j -= 1
            nums[j+1] = key
            
#for low and high use 0 and len of list - 1
def quick_sort(nums, low, high):
    if low < high:
        pivot = partition(nums, low, high)
        quick_sort(nums, low, pivot - 1)
        quick_sort(nums, pivot + 1, high)



def partition(nums, low, high):
    pivot = nums[high]
    i = low - 1
    for j in range(low, high):
        if nums[j] <= pivot:
            i += 1
            nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1], nums[high] = nums[high], nums[i+1]
    return i + 1


def heap_sort(nums):
    n = len(nums)
    for i in range(n//2-1, -1, -1):
        heapify(nums,n,i)
    for i in range(n-1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        heapify(nums, i , 0)

def heapify(nums, n, x):
    large = x
    l = 2*x+1
    r = 2*x+2
    if l < n and nums[large] < nums[l]:
        large  = l 
    if r < n and nums[large] < nums[r]:
        large = r
    if large != x:
        nums[x], nums[large] = nums[large], nums[x]
        heapify(nums, n, large)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("generating numbers....")
    numList =np.round(np.linspace(0,1000,20),0)
    np.random.seed(0)
    np.random.shuffle(numList)
    print(*numList)
    ans = True
    while ans:       
        display = True
        numList = TrackedArray(numList,"full")
        option = input("What sort would you like to view(use option to view avaliable sorts or q to quit): ")
        if option == "option":
            print("\n\nbubble(bubble sort), insert(insert sort), " 
                  "quick(quick sort), heap(heap sort)\n\n")
            display = False
        elif option == "bubble":
            bubble_sort(numList)
        elif option == "insert":
            insert_sort(numList)
        elif option == "merge":
            quick_sort(numList, 0, len(numList) -1)
        elif option == "heap":
            heap_sort(numList)
        elif option == 'q':
            ans = False
            display = False
        else:
            print("options not found")
            display = False
        if display:
            print(*numList)
            #plt.scatter(range(len(numList)), numList)
            fig, ex = plt.subplots()
            container = ex.bar(np.arange(0, len(numList), 1), numList.full_copies[0], align="edge", width = 0.8)
            ani = FuncAnimation(fig, update, frames=range(len(numList.full_copies)), blit = True, interval = 1000/60, repeat = False, fargs=(numList, container))
            plt.show()
           
        q = input("would you like to try again?(y or n): ")
        if q == 'y':
            numList = np.random.randint(0,1000, 20)
            print("numbers randomized")
            print(*numList)
        else:
            print("Thank you for using Vsort goodbye")
            ans = False


