#! /usr/bin/env python

import os,sys,re

if len(sys.argv)<3:
    print("usage: python score.py <prediction-file> <test-file>")
    sys.exit(2)

predictionFile = open(sys.argv[1],'r')
testDataFile = open(sys.argv[2],'r')

predictions = predictionFile.readlines()
testData = testDataFile.readlines()

predictionFile.close()
testDataFile.close()

folders = os.listdir("../test")

folders = sorted(folders)

correct = [[0 for i in range(15)] for j in range(15)]

output = open("output",'w')

for i in range(len(predictions)):
    guess = int(predictions[i].split()[0]) -1
    answer = int(testData[i].split()[0]) -1
    correct[answer][guess] = int(correct[answer][guess]) + 1

for col in folders:
    output.write("\t")
    output.write("".join(re.findall("[a-z]",str(col)))[0:7])
    print("\t",end="")
    print("".join(re.findall("[a-z]",str(col)))[0:7],end="")

output.write("\n")
print()

rowCount = 0

for row in correct:
    output.write("".join(re.findall("[a-z]",folders[rowCount]))[0:7])
    output.write("\t  ")
    print("".join(re.findall("[a-z]",folders[rowCount]))[0:7],end="")
    print("\t  ",end="")
    colCount = 0
    for col in row:
        if rowCount != colCount:
            output.write(" "+str(col))
            output.write("\t  ")
            print(" "+str(col),end="")
            print("\t  ",end="")
        else:
            output.write("|"+str(col)+"|")
            output.write("\t  ")
            print("|"+str(col)+"|",end="")
            print("\t  ",end="")
        colCount = colCount + 1
    output.write("\n")
    rowCount = rowCount+1
    print()

correctGuess = 0

for i in range(15):
    correctGuess = correctGuess + correct[i][i]

print("Accuracy = ",end="")
print((correctGuess*100.0)/len(testData))
output.write("Accuracy = ")
output.write(str((correctGuess*100.0)/len(testData)))

output.close()
