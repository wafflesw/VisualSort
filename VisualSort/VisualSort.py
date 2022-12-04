

import random
import numpy as np
import matplotlib.pyplot as plt


def bubble_sort(nums):
    swap = True
    while swap:
        swap = False
        for x in range(0, len(nums)-1):
            if nums[x] > nums[x+1]:
                swap = True
                nums[x], nums[x+1] = nums[x+1], nums[x]

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


            



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    numList = [random.randint(1, 100)]
    for x in range(99):
        x += 1
        numList.append(random.randint(1, 100))
    print(*numList)
    radix_sort(numList)
    print(*numList)
    plt.scatter(range(len(numList)), numList)
    plt.show()

