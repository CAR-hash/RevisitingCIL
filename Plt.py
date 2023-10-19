import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl



def main():
    fig, axs = plt.subplots()

    baseline = np.array([86.83, 87.57, 85.15, 81.42, 79.56, 76.73, 75.72, 74.12, 73.19, 73.3])

    #flip
    #axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([87.17, 87.66, 85.04, 81.34, 79.53, 76.81, 75.77, 74.16, 73.23, 73.38])-baseline, **{'marker': 'o'}, label='p=1')
    #axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([87.0, 87.57, 85.09, 81.34, 79.49, 76.67, 75.65, 74.1, 73.19, 73.3])-baseline, **{'marker': '2'}, label='p=0.3')
    #axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([87.0, 87.57, 85.04, 81.34, 79.59, 76.84, 75.77, 74.16, 73.31, 73.38])-baseline, **{'marker': '8'}, label='p=0.6')
    #axs.plot([0,1,2,3,4,5,6,7,8,9], baseline-baseline, **{'marker': 's'}, label='baseline')

    #rotation
    #axs.plot([0,1,2,3,4], np.array([85.83, 86.74, 84.15, 80.88, 78.89])-baseline[:5], **{'marker': 'o'}, label='d=180')
    #axs.plot([0,1,2,3,4,5], np.array([86.67, 87.24, 84.71, 81.0, 79.12, 76.48])-baseline[:6], **{'marker': '2'}, label='d=90')
    #axs.plot([0,1,2,3,4], np.array([86.67, 86.99, 84.82, 80.96, 79.02])-baseline[:5], **{'marker': '8'}, label='d=45')
    #axs.plot([0,1,2,3,4,5,6,7,8,9], baseline-baseline, **{'marker': 's'}, label='baseline')

    #crop
    #axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([85.5, 86.41, 83.76, 79.75, 77.56, 75.0, 74.12, 72.95, 71.97, 72.05])-baseline, **{'marker': 'o'}, label='scale=(0.8,1.0),ratio=(0.75,1.33)')
    #axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([87.5, 87.82, 84.87, 80.88, 79.16, 76.34, 75.41, 73.98, 73.12, 73.35])-baseline, **{'marker': '2'}, label='scale=(0.75, 1),ratio=(0.8, 1.2)')
    #axs.plot([0,1,2,3,4,5,6,7,8,9], baseline-baseline, **{'marker': 's'}, label='baseline')

    #photometric
    axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([86.0, 86.49, 84.04, 80.67, 77.89, 75.17, 74.19, 72.81, 71.73, 71.98])-baseline, **{'marker': 'o'}, label='gray scaling p=1')
    axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([86.5, 87.57, 85.26, 81.59, 79.49, 76.7, 75.7, 74.1, 73.19, 73.22])-baseline, **{'marker': '2'}, label='b=(0.6, 1.4), s=(0.6, 1.4)')
    axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([87.0, 87.66, 84.98, 81.34, 79.46, 76.7, 75.7, 74.06, 73.18, 73.25])-baseline, **{'marker': '4'}, label='b=(0.8, 1.2), s=(0.8, 1.2)')
    axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([87.33, 87.74, 85.21, 81.38, 79.69, 76.84, 75.79, 74.19, 73.25, 73.28])-baseline, **{'marker': '8'}, label='b=(0.8, 1.2), s=(0.75, 1.33)')
    axs.plot([0,1,2,3,4,5,6,7,8,9], np.array([87.17, 87.74, 85.09, 81.25, 79.46, 76.73, 75.67, 74.08, 73.16, 73.27])-baseline, **{'marker': '^'}, label='b=(0.75, 1.33), s=(0.8, 1.2)')
    axs.plot([0,1,2,3,4,5,6,7,8,9], baseline-baseline, **{'marker': 's'}, label='baseline')

    axs.set_xlabel("stage")
    axs.set_ylabel("Offset of average Top-1 accuracy")
    axs.grid(True)

    axs.legend()
    plt.show()

if __name__ == '__main__':
    main()