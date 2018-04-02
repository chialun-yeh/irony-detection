import csv

def readCsvFile(filename):

    testX=[]
    testY=[]

    with open(filename, encoding='utf-8') as csvDataFile:
	    csvReader = csv.reader(csvDataFile)
	    for row in csvReader:
		    testX.append(row[0])
		    testY.append(row[1])

    return testX, testY


# The following gives the testing text and label
# length: 1950
#textReddit, labelReddit = readCsvFile('../datasets/test/reddit/irony-labeled.csv')

