from model.DCGAN import *

if __name__ == '__main__':
    model = DCGAN()
    model.build()
    model.train()
    #model.restore()