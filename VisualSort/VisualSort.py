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

def tstart():
    t0 = time.perf_counter()
    return t0

def tstop(t0):
    return time.perf_counter() - t0

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
    for x in range(0,len(nums)-1):
        if nums[x] > nums[x+1]:
            temp = x
            nums[x], nums[x+1] = nums[x+1], nums[x]
            for j in range(x,-1,-1):
                if nums[j] > nums[temp]:
                    nums[j], nums[temp] = nums[temp], nums[j]
                    temp = j

def merge_sort(nums):
    if len(nums) > 1:
        mid = len(nums)//2
        L = nums[mid:]
        R = nums[:mid]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] >= R[j]:
                nums[k] = R[j]
                j += 1
            else:
                nums[k] = L[i]
                i += 1
            k += 1
        while i < len(L):
            nums[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            nums[k] = R[j]
            j += 1
            k += 1

#for low and high use 0 and len of array - 1
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


def count_sort(nums):
    max_element = int(max(nums))
    min_element = int(min(nums))
    range_of_elements = max_element - min_element + 1
    count_nums = [0 for i in range( range_of_elements )]
    output_nums = [0 for i in range( len(nums ))]
    for i in range(0, len(nums)):
        count_nums[nums[i]-min_element] += 1
    for i in range(1, len(count_nums)):
        count_nums[i] += count_nums[i-1]
    for i in range(len(nums)-1,-1,-1):
        output_nums[count_nums[nums[i] - min_element] - 1] = nums[i]
        count_nums[nums[i] - min_element] -= 1
    for i in range(0, len(nums)):
        nums[i] = output_nums[i]
    return nums

def bucket_sort(nums, n):
    min_ele = min(nums)
    max_ele = max(nums)
    rnge = (max_ele - min_ele)/n
    bucket = []
    for i in range(n):
        bucket.append([])
    for i in range(len(nums)):
        dif = (nums[i] - min_ele)/ rnge - int((nums[i] - min_ele)/rnge)
        if (dif == 0 and nums[i] != min_ele):
            bucket[int((nums[i] - min_ele)/rnge) - 1].append(nums[i])
        else:
            bucket[int((nums[i] - min_ele)/rnge)].append(nums[i])
    for i in range(len(bucket)):
        if len(bucket[i]) != 0:
            #uses the python integrated sort which is a timsort
            bucket[i].sort() 
    k = 0
    for lst in bucket:
        if lst:
            for i in lst:
                nums[k] = i
                k = k + 1

#this works by using the implemented counting sort and base 10 to compare each element 
def radix_sort(nums):
    max_ele = max(nums)
    exp = 1
    n = len(nums)
    while max_ele / exp >= 1:
        output = [0] * (n)
        count = [0] * (100)
        for i in range(0, n):
            index = nums[i] // exp
            count[index % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = nums[i] // exp
            output[count[index % 10] - 1] = nums[i]
            count[index % 10] -= 1
            i -= 1
        i = 0
        for i in range(0, n):
            nums[i] = output[i]
        exp *=10

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
    #numList = np.random.randint(0,1000, 20)
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
            print("bubble(bubble sort), insert(insert sort), merge(merge sort), \n" 
                  "quick(quick sort), count(count sort), bucket(bucket sort), \n" 
                  "radix(radix sort), heap(heap sort)")
            display = False
        elif option == "bubble":
            bubble_sort(numList)
        elif option == "insert":
            insert_sort(numList)
        elif option == "merge":
           merge_sort(numList)
        elif option == "quick":
            quick_sort(numList, 0, len(numList) -1)
        elif option == "count":
            count_sort(numList)
        elif option == "bucket":
            bucket_sort(numList,10)
        elif option == "radix":
            radix_sort(numList)
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


