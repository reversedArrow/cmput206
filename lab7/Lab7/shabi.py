import numpy as np




def getRegionalMinima(img):
    h,w = img.shape
    
    #init a all 0 metrix which has the same size
    new_img = np.zeros((h,w),dtype = np.int32)
    
    flag = 1
    
    for i in range(0,h):
        for j in range(0,w):
            localMin = True
            minlist = []
            
            #top left conrner
            if 0<=i-1 and 0<=j-1:
                minlist.append(img[i-1,j-1])
            #top border
            if 0<=i-1:
                minlist.append(img[i-1,j])
            #top right corner
            if 0<=i-1 and j+1<w:
                minlist.append(img[i-1,j+1])
            #bottom left corner
            if i+1<h and 0<=j-1:
                minlist.append(img[i+1,j-1])
            #bottom border
            if i+1<h:
                minlist.append(img[i+1,j])
            #bottom right corner
            if i+1<h and j+1<w:
                minlist.append(img[i+1,j+1])
            #left border
            if 0<=j-1:
                minlist.append(img[i,j-1])
            #right border  
            if j+1<w:
                minlist.append(img[i,j+1])            
                
            for value in minlist:
                if img[i,j] > value:
                    localMin = False
            if localMin:
                new_img[i,j] = flag
                flag += 1        
            #newimg fill with increased flag number at localMin location
        return new_img



def iterativeMinFollowing(img,markers):
    h,w = img.shape

    newimg = np.zeros((h,w),dtype = np.int32)
    
    count = 0
    
    for x in range(0,h):
        for y in range(0,w):
            if markers[x,y] == 0:
                count += 1
            else:              
                newimg[x,y] = markers[x,y]
                
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

                    pt = minlist[0]
                    for value in minlist:
                        if img[value] < img[pt]:
                            pt = value
                    if newimg[pt] != 0:
                        newimg[x,y] = newimg[pt]
                        count -= 1

    return newimg    