import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import math

color = {
    "Strežniška evalvacija": "gray",
    "1_huawei.csv": "blue",
    "2_pixel_6.csv": "orange",
    "3_pixel_7.csv": "green",
    "4_pixel_3.csv": "red",
    "5_pixel_8.csv": "purple",
}

markers = {
    "Strežniška evalvacija": "o",
    "1_huawei.csv": "s",
    "2_pixel_6.csv": "D",
    "3_pixel_7.csv": "x",
    "4_pixel_3.csv": "P",
    "5_pixel_8.csv": "h",
}

matplotlib.rcParams.update({'font.size': 13})

def draw_phone_num_graph(csvDir):
    # Read data from csv
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([0, 10, 0, 1])
    ax.locator_params(axis='x', integer=True)
    ax.set_xticks(range(0, 11, 1))
    for file in os.listdir(csvDir):
        if "bak" in file:
            continue
        with open(csvDir+"/"+file, 'r') as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1 if "server" in file else 0)
            x = data[:, 0]
            x = x.astype(int)
            if "server" in file:
                x = [ value for value in x]
            else:
                x = [ value+1 for value in x]
            y = data[:, 2]

            if "server" in file:
                file = "Strežniška evalvacija"
                label = "Strežniška evalvacija"
            else:
                label = " ".join(file.replace(".csv","").split("_")[1:]).capitalize()

            ax.plot(x, y, label = label, marker=markers[file], color=color[file])
            ax.set_xlabel('Št. runde')
            ax.set_ylabel('Natačnost')
            ax.legend()
    
    fig.savefig(f"{csvDir}.png")
    fig.clf()
    #plt.show()

def draw_epoch_num_graph(csvDir):
    # Read data from csv
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([1, 10, 0, 1])
    ax.set_xticks(range(0, 11, 1))
    for file in os.listdir(csvDir):
        if "bak" in file:
            continue
        with open(os.path.join(csvDir, file)) as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1 if "server" in file else 0)
            x = data[:, 0]
            if "server" in file:
                x = [ value for value in x]
            else:
                x = [ value+1 for value in x]
            y = data[:, 2]

            if "server" in file:
                file = "Strežniška evalvacija"
                label = "Strežniška evalvacija"
            else:
                label =  " ".join(file.replace(".csv","").split("_")[1:]).capitalize()

            ax.plot(x, y, label = label,marker=markers[file], color=color[file])
            ax.set_xlabel('Št. runde')
            ax.set_ylabel('Natančnost')
            ax.legend()
    
    fig.savefig(f"{csvDir}.png")
    fig.clf()

def draw_server_num_devices(csvDirList, measuring = "devices", start = 2,):
    # Read data from csv
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.locator_params(axis='x', integer=True)
    ax.axis([start, len(csvDirList)+start-1, 0.5, 0.8])
    maxAccList = []
    for csvDir in csvDirList:
        with open(os.path.join(csvDir, "server.csv")) as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1)
            if math.isnan(data[-1,0]):
                data = data[:-1]
            y = data[:, 2]
            maxAccList.append(max(y))

    print(maxAccList)
    ax.plot(range(start,len(csvDirList)+start, 1), maxAccList, label = "Strežniška evalvacija", marker=markers["Strežniška evalvacija"], color=color["Strežniška evalvacija"])
    if measuring == "devices":
        ax.set_xlabel('Št. naprav')
    else:
        ax.set_xlabel('Št. epoh')
    ax.set_ylabel('Natančnost')
    ax.legend()
    
    fig.savefig(f"experiments/server_num_{measuring}.png")
    fig.clf()

def draw_execution_time(csvDir):
    # Read data from csv
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([0, 10, 0, 3000])
    ax.set_xticks(range(0, 11, 1))
    for file in os.listdir(csvDir):
        if "server" in file or "bak" in file:
            continue
        with open(csvDir+"/"+file, 'r') as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1 if "server" in file else 0)
            x = data[:, 0]
            if "server" in file:
                x = [ value for value in x]
                label = "Strežniška evalvacija"
            else:
                x = [ value+1 for value in x]
                label =  " ".join(file.replace(".csv","").split("_")[1:]).capitalize()
            y = data[:, 1]
            
            ax.plot(x, y, label = label,marker=markers[file], color=color[file])
            ax.set_xlabel('Št. runde')
            ax.set_ylabel('Čas izvajanja runde (ms)')
            ax.legend()
    
    fig.savefig(f"{csvDir}-time.png")
    fig.clf()

def draw_server_total_execution_time(csvDirs):
    # Read data from csv
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([2, 5, 0, 60])
    ax.locator_params(axis='x', integer=True)
    totalTimes = []
    for file in csvDirs:
        if "bak" in file:
            continue
        with open(os.path.join(file,"server.csv"), 'r') as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1)
            y = data[-1, 2] / 1000
            totalTimes.append(y)
            
    ax.plot(range(2,6,1), totalTimes, label = "Strežniška evalvacija", marker=markers["Strežniška evalvacija"], color=color["Strežniška evalvacija"])
    ax.set_xlabel('Št. naprav')
    ax.set_ylabel('Skupen čas izvajanja (s)')
    ax.legend()
    
    fig.savefig(f"experiments/server-time.png")
    fig.clf()


if __name__ == "__main__":
    dirs = os.listdir(sys.argv[1])
    dirs = [ dir for dir in dirs if os.path.isdir(os.path.join(sys.argv[1], dir))]
    dirs = sorted(dirs)
    for dir in dirs:
        fullpath = os.path.join(sys.argv[1], dir)
        if not os.path.isdir(fullpath):
            continue
        if "num_devices" in fullpath:
            print(fullpath)
            draw_phone_num_graph(fullpath)
            draw_execution_time(fullpath)
        else:
            draw_epoch_num_graph(fullpath)
    numDevicesDirs =  [ os.path.join(sys.argv[1], dir) for dir in dirs if "num_devices" in dir]
    numEpochDirs =  [ os.path.join(sys.argv[1], dir) for dir in dirs if "num_epochs" in dir]
    print(numDevicesDirs)
    draw_server_num_devices(numDevicesDirs, "devices", start =2)
    draw_server_num_devices(numEpochDirs, "epochs", start = 5)
    draw_server_total_execution_time(numDevicesDirs)

