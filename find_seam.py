from show import *
from energy_function import *
from tqdm import tqdm, trange

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

# for "vertical seam":seam_orient='v', for "horizontal seam":seam_orient='h'
def find_1_seam(energy_map:np.ndarray, seam_orient:str) -> tuple([np.ndarray,float]):
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
    # smallest_k=np.array([(x,float('inf')) for x in range(seam_num)],dtype=[('index','<i4'),('total_energy','<f4')])
    smallest={'index':-1,'total_energy':float('inf')}
    if seam_orient == 'v':
        for j in range(width):
            min_e=min_E[height-1][j]
            if min_e < smallest.get('total_energy'):
                smallest['index'],smallest['total_energy']=j,min_e
        j=smallest['index']
        for i in reversed(range(height)):
            seam_map[i][j]=-1
            if min_O[i][j]==0:
                j-=1
            elif min_O[i][j]==2:
                j+=1
    elif seam_orient == 'h':
        for i in range(height):
            min_e = min_E[i][width-1]
            if min_e < smallest.get('total_energy'):
                smallest['index'],smallest['total_energy']=i,min_e
        i=smallest['index']
        for j in reversed(range(width)):
            seam_map[i][j]=-1
            if min_O[i][j]==0:
                i-=1
            elif min_O[i][j]==2:
                i+=1
    return seam_map, smallest['total_energy']

# def find_seam_2d(energy_map:np.ndarray, seam_num:int, seam_orient:str) -> np.ndarray:
    

def find_seam_2d_dp(_energy_map:np.ndarray, seam_horizontal:int, seam_vertical:int )->np.ndarray:
    # print(f"energy_map's type:{type(energy_map)}")
    T=np.zeros((seam_horizontal+1,seam_vertical+1))
    # C=np.zeros(T.shape)
    C=[[]for _ in range(seam_horizontal+1)]   #紀錄是走row還是col
    seam_M=[[]for _ in range(seam_horizontal+1)]   #紀錄seam_map
    energy_M=[[]for _ in range(seam_horizontal+1)]  #紀錄energy_map
    
    #應該初始化第一列
    T[0][0]=0
    energy_map=_energy_map.copy()
    seam_M[0].append(np.zeros(energy_map.shape))
    energy_M[0].append(energy_map)
    C[0].append('s')
    for c in range(1,seam_vertical+1):
        energy_map=energy_M[0][c-1]
        v_seam_map,E_v_seam=find_1_seam(energy_map,'v')
        T[0][c]=T[0][c-1]+E_v_seam
        seam_M[0].append(v_seam_map)
        energy_map = energy_map[~(v_seam_map == -1)].reshape(energy_map.shape[0], -1) 
        energy_M[0].append(energy_map)
        C[0].append('c')

    for r in tqdm(range(1,seam_horizontal+1),desc='horiaontal seam'):
        for c in tqdm(range(seam_vertical+1),desc='vertical seam'):
            if c == 0:
                energy_map=energy_M[r-1][0]
                h_seam_map,E_h_seam=find_1_seam(energy_map,'h')
                T[r][0]=T[r-1][0]+E_h_seam
                seam_M[r].append(h_seam_map)

                energy_map=energy_map.transpose()
                energy_map = energy_map[~(h_seam_map.transpose() == -1)].reshape(energy_map.shape[0], -1)
                energy_map=energy_map.transpose()
                energy_M[r].append(energy_map)
                C[r].append('r')
                continue
            h_seam_map,E_h_seam=find_1_seam(energy_M[r-1][c],'h')
            v_seam_map,E_v_seam=find_1_seam(energy_M[r][c-1],'v')
            if T[r-1][c]+E_h_seam <= T[r][c-1]+E_v_seam:
                T[r][c]=T[r-1][c]+E_h_seam
                ###
                C[r].append('r')
                seam_M[r].append(h_seam_map)
                energy_map=energy_M[r-1][c]
                # map_records.append(h_seam_map)

                # 假設是找 horizontal seam，代表是高度有縮減, 而寬度固定
                # energy_map = cv2.rotate(energy_map, cv2.ROTATE_90_CLOCKWISE)
                energy_map=energy_map.transpose()
                energy_map = energy_map[~(h_seam_map.transpose() == -1)].reshape(energy_map.shape[0], -1) 
                # energy_map = cv2.rotate(energy_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
                energy_map=energy_map.transpose()

            else:
                T[r][c]=T[r][c-1]+E_v_seam
                C[r].append('c')
                seam_M[r].append(v_seam_map)
                energy_map=energy_M[r][c-1]
                 # 反之為寬度有縮減, 而高度固定

                energy_map = energy_map[~(v_seam_map == -1)].reshape(energy_map.shape[0], -1)
                # map_records.append(v_seam_map)
            #如果C是'r',移除的map是根據r_map;else,移除的map是根據c_map
                 
            energy_M[r].append(energy_map)
    print("start tracing\n")
    map_records = []   #紀錄seam_map
    r,c=seam_horizontal,seam_vertical
    o='h'if C[r][c]=='r'else 'v'
    map_records.append((o,seam_M[r][c]))
    while r!=0 or c!=0:
        if o=='h':
            r-=1
            
        elif o=='v':
            c-=1
        o='h'if C[r][c]=='r'else 'v'
        map_records.insert(0,(o,seam_M[r][c]))
        # orient_records.insert(0,C)
    return map_records