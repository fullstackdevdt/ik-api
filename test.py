import array


def addTwoValues(a, b):
    return a + b

def addThreeValues(a,b,c):
    result = addTwoValues(a,b)
    finalResult = addTwoValues(result,c)
    return finalResult


def modifyArray(arr):
    for i in range(len(arr)):
        arr[i] = arr[i] * 2
    return arr

def arraySum(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total


# Define an empty dictionary
my_dict = {}

# Define a dictionary with initial key-value pairs
my_dict = {
    "name": "John",
    "age": 30,
    "city": "New York",
    2.1: ["ann", "griffin"]
}

print(my_dict[2.1])

my_dict["email"] = "john@example.com"

print(my_dict["email"])



# my_list = [1, 2, 3, 4, 5]

# my_array = array.array('i', [1, 2, 3, 4, 10])

# sum1 = arraySum(my_array)
# sum2 = sum(my_list)
# print("Sum of array elements:", sum1)
# print("Sum of list elements:", sum2)


# sum = addTwoValues(5,3)
# print("Sum of two values:", sum)

# sum2 = addThreeValues(2,4,6)
# print("Sum of three values:", sum2)