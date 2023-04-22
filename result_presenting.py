import numpy as np
import matplotlib.pyplot as plt
import os
LOGDIR="training_logs"

def main():
    traininfo=read_from_logfile("230422_1957.txt")
    # print(traininfo[:,0])
    plot_train_info(traininfo,
                    # in_range=(0,200)
                    )


def read_from_logfile(filepath):
    filepath=os.path.join(LOGDIR,filepath)
    train_info=[]
    with open(filepath,"r") as f:
        lines=f.readlines()
        for line in lines:
            item=line.split(',')
            if len(item)==5:
                train_info.append([float(item[i].split(':')[-1]) for i in range(5)])
                # print(item)
    train_info=np.array(train_info)
    return train_info

def plot_train_info(train_info,in_range=None,show=True,save=True,savefile=os.path.join(LOGDIR,"plots","tempfig.png")):
    fig=plt.figure(figsize=[10,8])
    ax1=fig.add_subplot(2,1,1)
    ax2=fig.add_subplot(2,1,2)
    if not in_range:
        epoch=train_info[:,0]
        TL=train_info[:,1]
        VL=train_info[:,2]
        TA=train_info[:,3]
        VA=train_info[:,4]
    else:
        epoch=train_info[in_range[0]:in_range[1],0]
        TL=train_info[in_range[0]:in_range[1],1]
        VL=train_info[in_range[0]:in_range[1],2]
        TA=train_info[in_range[0]:in_range[1],3]
        VA=train_info[in_range[0]:in_range[1],4]


    ax1.plot(epoch,TL,label='training loss')
    ax1.plot(epoch,VL,label='validation loss')

    ax2.plot(epoch,TA,label='training accuracy')
    ax2.plot(epoch,VA,label='validation accuracy')
    ax2.plot(epoch,np.ones(epoch.shape)*0.025,color='red',label='1/40')

    ax1.legend()
    ax2.legend()

    ax1.set_title("Losses")
    ax2.set_title("accuracies")
    
    if show:
        plt.show()
    
    if save:
        fig.savefig(savefile)



if __name__=="__main__":
    main()