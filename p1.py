"""
Katherine St. John
katherine.stjohn@hunter.cuny.edu
Program 1, Fall 2023
Resources:  Used CSci 127 textbook for files
"""
import textwrap
import random


def extract_overviews(file_name):
    """
    Opens the file_name and from each line of the file, keeps the overview
    description of the school (the fifth "column": overview_paragraph.
    Returns a list of the paragraphs.
    """

    lst = []
    #For each line:
    with open(file_name,encoding='UTF-8') as file_d:
        #throw away the header line:
        file_d.readline()
        for line in file_d:
            #Jump to URL:
            url_start = line.find('http')
            line = line[url_start+5:]
            start_quote = line.find(',"')
            line = line[start_quote+2:]
            end_quote = line.find('",')
            line = line[:end_quote]
            lst.append(line)
    return lst

def count_lengths(overview_list):
    """
    For each element of the overview_list, computes the length (# of characters).
    Returns the dictionary of length occurrences.
    """

    counts = {}
    for entry in overview_list:
        length = len(entry)
        counts[length] = counts.get(length,0) + 1
    return counts

def count_sentences(overview_list):
    """
    For each element of the overview_list, computes the number of periods
    (as a proxy for the number of sentences).
    Returns the dictionary of occurrences.
    """

    counts = {}
    for entry in overview_list:
        dots = entry.count('.')
        counts[dots] = counts.get(dots,0) + 1

    return counts


def compute_mean(counts):
    """
    Computes the mean of counts dictionary, weighting each key that occurs by its value.
    Returns the mean.
    """
    mean = 0

    num_values = sum(counts.values())
    total = sum([k*v for k,v in counts.items()])

    mean = total/num_values
    return mean


def compute_mse(theta, counts):
    """
    Computes the Mean Squared Error of the parameter theta and a dictionary, counts.
    Returns the MSE.
    """
    mse = 0

    num_values = sum(counts.values())
    total = sum([((k-theta)**2)*v for k,v in counts.items()])

    mse = total/num_values

    return mse

def test_compute_mean(mean_fnc=compute_mean):
    """
    Returns True if the mean_fnc performs correctly
    (e.g. computes weighted mean of inputted dictionary) and False otherwise.
    """

    #Should return num:
    num = random.randint(100,500)
    test0 = {num : 1}
    #Should also return num:
    test1 = {num : random.randint(1,10)}
    #Should return (5*100+4*200+400)/10 = 1700/10 = 170
    test2 = {100:5,200:4,400:1}

    if mean_fnc(test0) != num:
        correct = False
    elif mean_fnc(test1) != num:
        correct = False
    elif mean_fnc(test2) != 170:
        correct = False
    else:
        correct = True

    return correct


def test_mse(mse_fnc=compute_mse):
    """
    Returns True if the extract_fnc performs correctly
    (e.g. computes mean squared error) and False otherwise.
    """

    #Use with 0 and num as thetas
    num = random.randint(100,500)
    test0 = {num : 1}
    #Should also return num:
    count = random.randint(1,10)
    test1 = {num : count}
    #Should return for theta = 20, ((10-20)^2+2*0+(30-20)^2) = (100+0+100)/4 = 200/4 = 50
    test2 = {10:1,20:2,30:1}

    if mse_fnc(0,test0) != num*num:
        correct = False
        print("failed test 0")
    elif mse_fnc(num,test0) != 0:
        correct = False
    elif mse_fnc(0,test1) != num*num:
        correct = False
    elif mse_fnc(num,test1) != 0:
        correct = False
    elif mse_fnc(20,test2) != 50:
        correct = False
    else:
        correct = True
    print(correct)
    return correct

def test_count_lengths(counts_fnc=count_lengths):
    """
    Returns True if the counts_fnc performs correctly
    (e.g. counts the lengths of overviews and stores in dictionary) and False otherwise.
    """

    #Should return a {5:1}
    test0 = ["Hello"]
    #Should also return {5: count}:
    count = random.randint(1,10)
    test1 = test0*count
    #Should return {3:2, 4:1, 10:1}
    test2 = ["bat","cat","cats","catepillar"]

    correct = True
    if counts_fnc(test0) != {5:1}:
        correct = False
    elif counts_fnc(test1) != {5: count}:
        correct = False
    else:
        result = counts_fnc(test2)
        answer = {3:2, 4:1, 10:1}
        for key,val in answer.items():
            if not key in result:
                correct = False
            elif result[key] != val:
                correct = False

    return correct








def main():
    """
    Some examples of the functions in use:
    """

    ###Extracts the overviews from the data files:
    file_name = 'fall23/program01/2021_DOE_High_School_Directory_SI.csv'
    si_overviews = extract_overviews(file_name)
    print(f"Number of SI overviews: {len(si_overviews)}. The the last one is:\n")
    #Using textwrap for prettier printing:
    print(textwrap.fill(si_overviews[-1],80))

    late_name = 'fall23/program01/2020_DOE_High_School_Directory_late_start.csv'
    late_overviews = extract_overviews(late_name)
    print(f"\n\nNumber of late start overviews: {len(late_overviews)}. The the last one is:\n")
    print(textwrap.fill(late_overviews[-1],80))

    ###Computing counts and means:
    si_len_counts = count_lengths(si_overviews)
    print(f"The {sum(si_len_counts.values())} entries have lengths:")
    print(si_len_counts)
    late_len_counts = count_lengths(late_overviews)
    print(f"The {sum(late_len_counts.values())} entries have lengths:")
    print(late_len_counts)

    si_dots_counts = count_sentences(si_overviews)
    print(f"The {sum(si_dots_counts.values())} entries have lengths:")
    print(si_dots_counts)
    late_dots_counts = count_sentences(late_overviews)
    print(f"The {sum(late_dots_counts.values())} entries have lengths:")
    print(late_dots_counts)

    si_len_mean = compute_mean(si_len_counts)
    si_dots_mean = compute_mean(si_dots_counts)
    print(f"Staten Island high schools overviews had an average of {si_len_mean:.2f}\
 characters in {si_dots_mean:.2f} sentences.")

    ###Computing MSE:
    late_dots_mean = compute_mean(late_dots_counts)
    print(f"The mean for number of sentences in SI descriptions is {late_dots_mean}.")
    losses = []
    for theta in range(10):
        loss = compute_mse(theta,late_dots_counts)
        print(f"For theta = {theta}, MSE loss is {loss:.2f}.")
        losses.append(loss)

    losses = []
    for theta in range(10):
        loss = compute_mse(theta,si_dots_counts)
        print(f"For theta = {theta}, MSE loss is {loss:.2f}.")
        losses.append(loss)

    ###Testing
    #Trying first on the correct function:
    print(f'test_compute_mean(compute_mean) returns {test_compute_mean(compute_mean)}.')
    #Trying on a function that returns 42 no matter what the output:
    print(f'test_compute_mean( lambda x : 42 ) returns {test_compute_mean(lambda x : 42)}.')



if __name__ == "__main__":
    main()
