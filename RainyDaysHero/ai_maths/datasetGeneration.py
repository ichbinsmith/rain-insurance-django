import csv
import os
import itertools

os.chdir(os.path.abspath(os.path.dirname(__file__)))

from premiumComputation import TermInsuranceAnnual, WholeLifeAnnual,PEAnnual

def generateTermInsuranceDataset():
	age = [20, 30, 40, 50, 60]
	payements = [1, 5, 10, 20, 30, 40]
	maturity = [1, 5, 10, 20, 30, 40]
	interest_rate = [0, 0.005, 0.01, 0.015, 0.02, 0.025]
	amount = [500, 1000, 2000, 4000, 8000]

	data_parameters = tuple(map(tuple, itertools.product(age, payements, maturity, interest_rate, amount)))
	data_parameters = list(filter(lambda x: x[1]<=x[2],data_parameters))

	nb_parameters_set = len(data_parameters)

	for i in range(nb_parameters_set):
	    data_parameters[i] = data_parameters[i] + (TermInsuranceAnnual(*data_parameters[i]), )

	data_file = 'dataset.csv'
	first_line = ('age', 'nb_payements', 'maturity', 'interest_rate', 'amount', 'target')

	if not (os.path.exists(data_file) and os.path.getsize(data_file) > 0):
	    with open(data_file, 'w', newline='') as file:
	        writer = csv.writer(file, delimiter=',')
	        writer.writerow(first_line)

	    for data in data_parameters:
	        with open(data_file, 'a', newline='') as file:
	                writer = csv.writer(file, delimiter=',')
	                writer.writerow(data)


def generatePureEndowmentDataset():
	age = [20, 30, 40, 50, 60]
	payements = [1, 5, 10, 20, 30, 40]
	maturity = [1, 5, 10, 20, 30, 40]
	interest_rate = [0, 0.005, 0.01, 0.015, 0.02, 0.025]
	amount = [500, 1000, 2000, 4000, 8000]

	data_parameters = tuple(map(tuple, itertools.product(age, payements, maturity, interest_rate, amount)))
	data_parameters = list(filter(lambda x: x[1]<=x[2],data_parameters))

	nb_parameters_set = len(data_parameters)

	for i in range(nb_parameters_set):
	    data_parameters[i] = data_parameters[i] + (PEAnnual(*data_parameters[i]), )

	data_file = 'dataset.csv'
	first_line = ('age', 'nb_payements', 'maturity', 'interest_rate', 'amount', 'target')

	if not (os.path.exists(data_file) and os.path.getsize(data_file) > 0):
	    with open(data_file, 'w', newline='') as file:
	        writer = csv.writer(file, delimiter=',')
	        writer.writerow(first_line)

	    for data in data_parameters:
	        with open(data_file, 'a', newline='') as file:
	                writer = csv.writer(file, delimiter=',')
	                writer.writerow(data)

def generateWholeLifeDataset():
	age = [20, 30, 40, 50, 60]
	payements = [1, 5, 10, 20, 30, 40]
	interest_rate = [0, 0.005, 0.01, 0.015, 0.02, 0.025]
	amount = [500, 1000, 2000, 4000, 8000]

	data_parameters = list(map(tuple, itertools.product(age, payements, interest_rate, amount)))
	#data_parameters = list(filter(lambda x: x[1]<=x[2],data_parameters))

	nb_parameters_set = len(data_parameters)

	for i in range(nb_parameters_set):
	    data_parameters[i] = data_parameters[i] + (WholeLifeAnnual(*data_parameters[i]), )

	data_file = 'dataset.csv'
	first_line = ('age', 'nb_payements','interest_rate', 'amount', 'target')

	if not (os.path.exists(data_file) and os.path.getsize(data_file) > 0):
	    with open(data_file, 'w', newline='') as file:
	        writer = csv.writer(file, delimiter=',')
	        writer.writerow(first_line)

	    for data in data_parameters:
	        with open(data_file, 'a', newline='') as file:
	                writer = csv.writer(file, delimiter=',')
	                writer.writerow(data)



def generateLifeAnnuitiesDataset():
	age = [20, 30, 40, 50, 60]
	payements = [1, 5, 10, 20, 30, 40]
	maturity = [1, 5, 10, 20, 30, 40]
	interest_rate = [0, 0.005, 0.01, 0.015, 0.02, 0.025]
	amount = [500, 1000, 2000, 4000, 8000]

	data_parameters = tuple(map(tuple, itertools.product(age, payements, maturity, interest_rate, amount)))
	data_parameters = list(filter(lambda x: x[1]<=x[2],data_parameters))

	nb_parameters_set = len(data_parameters)

	for i in range(nb_parameters_set):
	    data_parameters[i] = data_parameters[i] + (LAAnnual(*data_parameters[i]), )

	data_file = 'dataset.csv'
	first_line = ('age', 'nb_payements', 'maturity', 'interest_rate', 'amount', 'target')

	if not (os.path.exists(data_file) and os.path.getsize(data_file) > 0):
	    with open(data_file, 'w', newline='') as file:
	        writer = csv.writer(file, delimiter=',')
	        writer.writerow(first_line)

	    for data in data_parameters:
	        with open(data_file, 'a', newline='') as file:
	                writer = csv.writer(file, delimiter=',')
	                writer.writerow(data)

def main():
	generateWholeLifeDataset()

if __name__ == '__main__':
	main()