import numpy as np

def getRegionalMinima(img):
    # add your code here
    h,w = img.shape

    #init a all 0 metrix which has the same size
    newimg = np.zeros((h,w),dtype = np.int32)
    flow = 1
    for x in range(0,h):
        for y in range(0,w):
            localmin = True
            minlist = []
            #top left czonrner
            if 0<=x-1 and 0<=y-1:
                minlist.append(img[x-1,y-1])
            #top border
            if 0<=x-1:
                minlist.append(img[x-1,y])
            #top right corner
            if 0<=x-1 and y+1<w:
                minlist.append(img[x-1,y+1])
            #bottom left corner
            if x+1<h and 0<=y-1:
                minlist.append(img[x+1,y-1])
            if x+1<h:
                minlist.append(img[x+1,y])
            #bottom right corner
            if x+1<h and y+1<w:
                minlist.append(img[x+1,y+1])
            #left border
            if 0<=y-1:
                minlist.append(img[x,y-1])
            #right border  
            if y+1<w:
                minlist.append(img[x,y+1])

            # #newimg fill with increased flag number at localMin location
            for coord in minlist:
                if img[x,y] > coord:
                    localmin = False
            if localmin:
                newimg[x,y] = flow
                flow += 1
                        
    return newimg



def iterativeMinFollowing(img, markers):
    # add your code here
    h,w = img.shape

    newimg = np.zeros((h,w),dtype = np.int32)
    count = 0
    
    for x in range(0,h):
        for y in range(0,w):
            if markers[x,y] == 0:
                count += 1
            else:              
                newimg[x,y] = markers[x,y]#If p is already labeled (i.e. it has a non zero value), leave it unchanged
    #print count
    while count>0:
        for x in range(0,h):
            for y in range(0,w):
                if newimg[x,y] == 0:
                    minlist = []
                    #top left conrner
                    if 0<=x-1 and 0<=y-1:
                        minlist.append((x-1,y-1))
                    #top border 
                    if 0<=x-1:
                        minlist.append((x-1,y))
                    #top right corner
                    if 0<=x-1 and y+1<w:
                        minlist.append((x-1,y+1))
                    #bottom left corner
                    if x+1<h and 0<=y-1:
                        minlist.append((x+1,y-1))
                    #bottom border
                    if x+1<h:
                        minlist.append((x+1,y))
                    #bottom right corner
                    if x+1<h and y+1<w:
                        minlist.append((x+1,y+1))
                    #left border
                    if 0<=y-1:
                        minlist.append((x,y-1))
                    #right border
                    if y+1<w:
                        minlist.append((x,y+1))
                    # Otherwise, find the pixel with the smallest intensity value in the 8 connected neighborhood of p
                    # If the smallest neighbor has a non zero label, mark p with its label; otherwise, leave it unchanged
                    nb = minlist[0]
                    for coord in minlist:
                        if img[coord] < img[nb]:
                            nb = coord
                    if newimg[nb] != 0:
                        newimg[x,y] = newimg[nb]
                        count -= 1

    return newimg