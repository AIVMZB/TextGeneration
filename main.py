from model import test_model

if __name__ == '__main__':
    text = input(">>>")
    result = test_model(text, 25)

    print(result)
