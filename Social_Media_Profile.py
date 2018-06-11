import Emotional_GloVe


if __name__ == '__main__':
    sentences = ["I am so sick of this", "God Damn", "this is kinda cute", "happyyyy", "This is fine,i am so grateful"]
    Res = Emotional_GloVe.main(input=sentences)

    print(Res)