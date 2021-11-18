from pysimme import RevisedSimplexMethod


# Case 1
A = [[6,   5, 1, 0, 0],
     [10, 20, 0, 1, 0],
     [1,   0, 0, 0, 1]]
c = [500, 450, 0, 0, 0]
b = [60, 150, 8]

model = RevisedSimplexMethod(c, A, b, mini=False)
model.compute()

sol, obj = model.get_optimal()
print('solution:', sol)
print('object:', obj)
model.summary(outfile='report_case1.txt', dec=3)


# Case 2
A = [[5, -4, 13, -2, 1],
     [1, -1,  5, -1, 1]]
c = [3, -1, -7, 3, 1]
b = [20, 8]
model = RevisedSimplexMethod(c, A, b, mini=True)
model.compute()
model.summary(outfile='report_case2.txt', dec=3)