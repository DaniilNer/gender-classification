def findMaxSubArray(A):
    max_sum = float('-inf')
    sub_left = None
    sub_right = None
    summ = 0
    for i in range(len(A)):
        if summ > 0: summ += A[i]
        else:
            left = i
            summ = A[i]
        if summ > max_sum:
            max_sum = summ
            sub_left = left
            sub_right = i + 1
    return A[sub_left : sub_right]


print(findMaxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
