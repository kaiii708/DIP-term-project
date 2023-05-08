import numpy as np

# for "vertical seam":seam_orient='v', for "horizontal seam":seam_orient='h'
def find_seam(energy_map:np.ndarray, seam_num:int, seam_orient:str) -> np.ndarray:
    height,width = energy_map.shape
    min_E = np.zeros((height,width))
    min_O = np.zeros((height,width))
    seam_map = np.zeros((height,width))
    if seam_orient == 'v':
        min_E[0][:]=energy_map[0][:]
        min_O[0][:]=-1
        for i in range(1,height):
            for j in range(0,width):
                e=energy_map[i][j]
                ancestor = {}
                if j == 0:
                    ancestor[1]=min_E[i-1][j]
                    ancestor[2]=min_E[i-1][j+1]
                elif j == width-1:
                    ancestor[0]=min_E[i-1][j-1]
                    ancestor[1]=min_E[i-1][j]
                else:
                    ancestor[0]=min_E[i-1][j-1]
                    ancestor[1]=min_E[i-1][j]
                    ancestor[2]=min_E[i-1][j+1]
                min_E[i][j]=min(ancestor.values())+e
                min_O[i][j]=min(ancestor,key=ancestor.get)
    elif seam_orient == 'h':
        min_E.transpose()[0][:]=energy_map.transpose()[0][:]
        min_O.transpose()[0][:]=-1
        for j in range(1,width):
            for i in range(0,height):
                e=energy_map[i][j]
                ancestor = {}
                if i == 0:
                    ancestor[1]=min_E[i][j-1]
                    ancestor[2]=min_E[i+1][j-1]
                elif i == height-1:
                    ancestor[0]=min_E[i-1][j-1]
                    ancestor[1]=min_E[i][j-1]
                else:
                    ancestor[0]=min_E[i-1][j-1]
                    ancestor[1]=min_E[i][j-1]
                    ancestor[2]=min_E[i+1][j-1]
                min_E[i][j]=min(ancestor.values())+e
                min_O[i][j]=min(ancestor,key=ancestor.get)
    smallest_k=np.array([(x,float('inf')) for x in range(seam_num)],dtype=[('index','<i4'),('total_energy','<f4')])
    if seam_orient == 'v':
        for j in range(width):
            for k in range(seam_num):
                if min_E[height-1][j] < smallest_k['total_energy'][k]:
                    smallest_k[k]=(j,min_E[height-1][j])
                    break
        for k in range(seam_num):
            j=smallest_k['index'][k]
            for i in reversed(range(height)):
                seam_map[i][j]=-1
                if min_O[i][j]==0:
                    j-=1
                elif min_O[i][j]==2:
                    j+=1
    elif seam_orient == 'h':
        for i in range(height):
            for k in range(seam_num):
                if min_E[i][width-1] < smallest_k['total_energy'][k]:
                    smallest_k[k]=(i,min_E[i][width-1])
                    break
        for k in range(seam_num):
            i=smallest_k['index'][k]
            for j in reversed(range(width)):
                seam_map[i][j]=-1
                if min_O[i][j]==0:
                    i-=1
                elif min_O[i][j]==2:
                    i+=1
    return seam_map
