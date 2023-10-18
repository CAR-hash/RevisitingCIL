import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def main():
    fig, axs = plt.subplots()



    axs.plot([1,2,3,4], [1,4,2,3], label='p=1')
    axs.xlabel()

    axs.legend()
    plt.show()

if __name__ == '__main__':
    main()