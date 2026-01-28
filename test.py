


def addTwoValues(a, b):
    return a + b

def addThreeValues(a,b,c):
    result = addTwoValues(a,b)
    finalResult = addTwoValues(result,c)
    return finalResult


sum = addTwoValues(5,3)
print("Sum of two values:", sum)

sum2 = addThreeValues(2,4,6)
print("Sum of three values:", sum2)