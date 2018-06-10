
def main():

    # Hyper Param
    Seq_len = 100

    # Step 1 : Load the book
    ebook = open("wonderland.txt").read().lower()

    # Step 2 : Convert the book to a char seq represented in integers
    distinct_chars_list = sorted(list(set(ebook)))
    char_dic = dict((char, index) for index, char in enumerate(distinct_chars_list))

    total_chars = len(ebook)
    total_vocab = len(distinct_chars_list)

    # Step 3 : Retrieve each time prediction pattern
    # list of list of patters for each step in the whole ebook
    input_patterns = []

    # list of predicted char for each step in the whole ebook
    output_chars = []

    num_of_patterns = total_chars - Seq_len

    for i in range(0, num_of_patterns):
        # for each char in the range from i to i+Seq_len in the ebook , get its integer and append it to the list
        input_patterns.append([char_dic[char] for char in ebook[i:i+Seq_len]])

        output_chars.append(char_dic[ebook[i+Seq_len]])




    return

if __name__ == '__main__':
    main()
