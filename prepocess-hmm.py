import pandas as pd
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from hmmlearn.hmm import MultinomialHMM

import warnings
warnings.filterwarnings("ignore")


def read_raw_data(fpath = "./data/splited_chengdushi_1001_1010_0.csv",row_number=4000):
    #parameters
    keep_every = 20
    # 每格长度(m)
    xstep =250
    ystep = 250
    time_slice = 20 * 60  # seconds

    print(f'Reading data: {row_number:d} rows.')

    # 读数据到dataframe，去掉重复的列
    df = pd.read_csv(fpath, nrows=row_number,names=['_', 'POLYLINE','__'], header=1).drop_duplicates()  #(2140172, 3)
    print(df.head())
    #df = df.fillnan()
    # 删除前两列
    df = df.drop(['_'], axis=1).drop(['__'], axis=1)

    # 删除为空的数据
    df = df[~df['POLYLINE'].str.contains("\[,\]|\[\]")]

    # 重新排序
    # df = df.sample(frac=1).reset_index(drop=True)

    # 插入标号列
    df.insert(0, "id", list(range(df.size)), allow_duplicates=False)

    # 分割每个点
    df['POLYLINE'] = (df['POLYLINE'].str.slice(1, -1).str.split(","))
    # 每个点变单行
    df = df.explode('POLYLINE')

    # 分开lng,lat,timestamp
    df[['lng', 'lat', 'timestamp']] = df['POLYLINE'].str.extract(
        '([+-]?\d+[\\.]?\d*) ([+-]?\d+[\\.]?\d*) (\d+)', expand=True)
    df = df.drop(["POLYLINE"], axis=1).reset_index(drop=True)

    # 筛除部分点
    df = df.iloc[df.index[df.index % keep_every == 0],:].reset_index(drop=True)

    # 格式转换
    df['lng'] = df['lng'].astype(np.float64)
    df['lat'] = df['lat'].astype(np.float64)
    #df = df.fillna(0)
    print(df.head())
    df['timestamp'] = df['timestamp'].astype(np.int)
    # for i in range(0,df.shape[0]):
    #     if df['timestamp'][i]==df['timestamp'][i]:
    #         df['timestamp'][i]=0
    #     else:
    #         df['timestamp'][i] = df['timestamp'][i].astype(np.int)
    # df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # delete points that have the same timestamp
    df = df[~df.duplicated(subset=['id','timestamp'],keep='first')]

    # 每条轨迹上的点按时间排序
    df = df.sort_values(by=['id', 'timestamp'], axis=0)

    # 经纬度转米的函数
    def lon2metersx(lon):  # x经度
        semimajoraxis = 6378137.0
        east = lon * 0.017453292519943295
        return semimajoraxis * east

    def lat2metersy(lat):  # y 纬度
        north = lat * 0.017453292519943295
        t = math.sin(north)
        return 3189068.5 * math.log((1 + t) / (1 - t))

    # 准备划分网格函数的参数
    df['x0'] = df['lng'].apply(lambda x: lon2metersx(x))
    df['y0'] = df['lat'].apply(lambda x: lat2metersy(x))

    xmin = df['x0'].min()
    xmax = df['x0'].max()
    ymin = df['y0'].min()
    ymax = df['y0'].max()
    x_cells = int((xmax - xmin) / xstep)
    y_cells = int((ymax - ymin) / ystep)

    print(f"Longtitude cells:{x_cells}, latitude cells:{y_cells}, total cells:{x_cells*y_cells}.")
    # Longtitude cells:96, latitude cells:95, total cells:9120.

    # 按米来编号，从0开始到x_cells-1
    def __funx(x):
        i = int((x - xmin) / xstep)
        if (i != x_cells):
            return i
        else:
            return x_cells - 1

    # 按米来编号，从0开始到y_cells-1
    def __funy(x):
        i = int((x - ymin) / ystep)
        if (i != y_cells):
            return i
        else:
            return y_cells - 1

    # 得到行列编号
    df['x'] = df['x0'].apply(lambda x: __funx(x))
    df['y'] = df['y0'].apply(lambda x: __funy(x))

    # 得到总编号
    df['number'] = df['x'] * y_cells + df['y']

    # delete continous points that are in the same cell
    df = df[~(df.duplicated(subset=['id', 'number'], keep='first') & (df['number'].diff(1) == 0))]

    # 保留需要的列，并调整顺序
    df = df[['id','timestamp', 'x0', 'y0', 'number']]

    # devide time slice
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    total_hours = int((max_time-min_time)/time_slice) + 1

    Trjss = [[] for x in range(total_hours)]
    for id in df['id'].unique():
        temp = df[df['id'] == id]
        # 抛弃少于10个点或大于100个点的序列
        if temp.shape[0] < 10 or temp.shape[0] > 100:
            continue
        start_time = temp['timestamp'].iloc[0]
        temp['relative_time'] = temp['timestamp'] - start_time
        # ['timestamp', 'x0', 'y0', 'number', 'relative_time']
        Trjs = temp.iloc[:,1:].values.tolist()  # (N,5)
        Trjss[int((start_time-min_time)/time_slice)].append(Trjs)

    # sorted_trjss = []
    # for Trjs in Trjss:
    #     if len(Trjs) < 10:
    #         continue
    #     else:
    #         sorted_trjss.append(sorted(Trjs, key=lambda x:x[0][0]))

    #model = MultinomialHMM(n_components=5)  # n_components=5, covariance_type="diag", n_iter=1000)
    Trjss = [sorted(x, key=lambda x:x[0][0]) for x in Trjss]
    test_set=Trjss[len(Trjss)*4//5:]
    Trjss=Trjss[:len(Trjss)*4//5]

    x = 250

    Thmm=[]
    #model = MultinomialHMM(n_components=5)  # 必须每次重新初始化
    for t in Trjss:
        for ts in t:
            hmms=[]
            for tss in ts:
                hmms.append(tss[3])
            hmms = [[int(e), ] for e in hmms]
            for e in hmms:
                Thmm.append(e)
    Testhmm = []
    # model = MultinomialHMM(n_components=5)  # 必须每次重新初始化
    for t in test_set:
        for ts in t:
            hmms = []
            for tss in ts:
                hmms.append(tss[3])
            Testhmm.append(hmms)


    model = MultinomialHMM(n_components=5)  # 必须每次重新初始化
    model.fit(Thmm)
    num_samples = 32
    sum=int(len(Testhmm))
    fdata=[]
    for i in range (0,sum):
        samples, _ = model.sample(num_samples)
        sampless = [int(e)  for e in samples]
        #tuple_lists = [[i[0]//x,i[0]%x]for i in samples]
        fdata.append(sampless)


    pickle.dump(fdata, open('./data/fdata', 'wb+'))
    pickle.dump(Testhmm, open('./data/Testhmm', 'wb+'))


    #samples, _ = model.sample(num_samples)
    #samples2, _ = model.sample(num_samples)

    # for t in Trjss:
    #     for ts in t:
    #         hmms=[]
    #         for tss in ts:
    #             hmms.append(tss[3])
    #         hmms = [[int(e), ] for e in hmms]
    #         #o=hmms.max()
    #         model = MultinomialHMM(n_components=5)#必须每次重新初始化
    #         model.fit(hmms)
    #         num_samples = 1
    #         samples, _ = model.sample(num_samples)
    #         samples2, _ = model.sample(num_samples)
    #         hmms.append([int(samples[0])])
    #         Thmm.append(hmms)

    pickle.dump(Trjss, open('./data/trajectories', 'wb'))
    print('Trajectories saved.')


# 计算一级特征
def completeTrajectories():
    # 轨迹集
    slice_data = pickle.load(open('./data/trajectories', 'rb'))
    slice_comps = []
    # 遍历每条轨迹
    for time_slice in slice_data:
        # 一级特征轨迹集
        traj_comps = []
        for traj in time_slice:
            # 每条轨迹的一级特征
            trjsCom = []
            for i in range(0, len(traj)):
                rec = []
                for j in range(0, 5):
                    # ['timestamp', 'x0', 'y0', 'number', 'relative_time']
                    rec.append(traj[i][j])
                # 时间差 Δt
                if i == 0:
                    rec.append(0)  # 第一个时间没有时间差，记为0
                else:
                    rec.append(traj[i][0] - traj[i - 1][0])  # 时间差
                # 相邻点位置变化（change of position）Δl
                locC = math.sqrt((traj[i][1] - traj[i - 1][1]) ** 2 + (traj[i][2] - traj[i - 1][2]) ** 2)
                rec.append(locC)
                # if simTrjs[i][0] - simTrjs[i - 1][0] < 1e-6:
                #     print('stop at:', i, "\n", simTrjs)
                if i == 0:
                    rec.append(0)  # 第一个时间没有时间差，记为0
                else:
                    rec.append(locC / (traj[i][0] - traj[i - 1][0]))  # 速度s
                # 旋转率（ROT，rate of turn）r
                if traj[i][1] - traj[i - 1][1] < 1e-6:
                    if traj[i][2] > traj[i - 1][2]:
                        rec.append(math.pi / 2)
                    else:
                        rec.append(-math.pi / 2)
                else:
                    rec.append(math.atan((traj[i][2] - traj[i - 1][2]) / (traj[i][1] - traj[i - 1][1])))
                if i == 0:
                    rec.append(0)  # 第一个没有差，记为0
                    rec.append(0)
                else:
                    rec.append(traj[i][1] - traj[i - 1][1])  # lng差
                    rec.append(traj[i][2] - traj[i - 1][2])  # lat差
                # [时间戳，经度转米，维度转米，网格号，轨迹历经时间，时间差，位置变化，速度7，旋转率8，横方向变化，纵方向变化]
                trjsCom.append(rec)  # (1, 11)
            traj_comps.append(trjsCom)
        slice_comps.append(traj_comps)
    pickle.dump(slice_comps, open('./data/trajectories_complete', 'wb'))
    print('Trajectories complete saved.')


# 计算二级特征
def computeFeas():
    slice_data = pickle.load(open('./data/trajectories_complete', 'rb'))
    slice_feas = []

    for traj_compss in slice_data:
        traj_feas = []
        for traj_comps in traj_compss:
            traj_com_fea = []
            for i in range(0, len(traj_comps)):
                rec = []
                for j in range(0, 11):
                    rec.append(traj_comps[i][j])
                if i == 0:
                    rec.append(0)  # 第一个没有差，记为0
                    rec.append(0)
                else:
                    rec.append(traj_comps[i][7] - traj_comps[i - 1][7])  # 速度变化（change of speed），Δs
                    rec.append(traj_comps[i][8] - traj_comps[i - 1][8])  # 旋转率变化（change of rot），Δr
                # [时间戳0，经度转米，维度转米，网格号
                # 轨迹历经时间4，时间差，位置变化，速度7，
                # 旋转率8，横方向变化，纵方向变化，速度变化，旋转率变化12]
                # 把网格编号调整到第一个
                rec = rec[3:4]+rec[0:3]+rec[4:]
                traj_com_fea.append(rec)  # (1, 13)
            traj_feas.append(traj_com_fea)
        slice_feas.append(traj_feas)

    # [网格号0，时间戳，经度转米，维度转米，
    # 轨迹历经时间4，时间差，位置变化，速度7，
    # 旋转率8，横方向变化，纵方向变化，速度变化，旋转率变化12]
    pickle.dump(slice_feas, open('./data/trajectories_feas', 'wb'))
    print('Trajectories features saved.')


# 先拼成一维，传入正则化函数，再还原成原来的维度
def generate_normal_behavior_sequence():
    f = open('./data/trajectories_feas', 'rb')
    slice_sequences = pickle.load(f)
    min_max_scaler = preprocessing.MinMaxScaler()

    slice_sequences_normal = []
    for behavior_sequences in slice_sequences:
        behavior_sequences_normal = []
        templist = []
        for item in behavior_sequences:
            for ii in item:
                ii = ii[1:]
                templist.append(ii)
        templist_normal = min_max_scaler.fit_transform(templist).tolist()
        index = 0
        for item in behavior_sequences:
            behavior_sequence_normal = []
            for ii in item:
                templist_normal[index] = ii[:1] + templist_normal[index]
                behavior_sequence_normal.append(templist_normal[index])
                index = index + 1
            behavior_sequences_normal.append(behavior_sequence_normal)
        slice_sequences_normal.append(behavior_sequences_normal)
    fout = open('./data/normal_feas', 'wb')
    pickle.dump(slice_sequences_normal, fout)
    print('Normalized features saved.')


# def get_start_and_end():
#     Trjss = pickle.load(open('./data/trajectories', 'rb'))
#     starts = []
#     ends = []
#     for Trjs in Trjss:
#         starts.append(Trjs[0][1:])
#         ends.append(Trjs[-1][1:])
#     starts = np.array(starts)
#     ends = np.array(ends)
#     for e in np.linspace(0.004, 0.006, 21):
#         for s in np.linspace(100, 300, 21):
#             DBSCAN_and_plot(starts, e, s, test=True)
#
#     # DBSCAN_and_plot(starts, 0.005, 300)
#
# def DBSCAN_and_plot(X, eps, min_samples, test=False):
#     db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#
#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#
#     if test:
#         if (n_clusters_<10):
#             return
#         # else:
#         #     print(f'eps: {eps:.4f} min_samples: {min_samples:.1f} n_clusters_: {n_clusters_:.1f}')
#         #     return
#
#     # Black removed and is used for noise instead.
#     unique_labels = set(labels)
#     colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#     for k, col in zip(unique_labels, colors):
#         if k == -1:
#             # Black used for noise.
#             col = 'k'
#
#         class_member_mask = (labels == k)
#
#         xy = X[class_member_mask & core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#                  markeredgecolor='k', markersize=7)
#
#         xy = X[class_member_mask & ~core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#                  markeredgecolor='k', markersize=3)
#
#     plt.title('Estimated clusters: %d' % n_clusters_)
#     plt.plot()
#     plt.savefig(f'./figures/eps{eps*10000:.0f}min_samples{min_samples:.0f}.png')
#
# def plot_starts_and_ends(starts, ends):
#     plt.subplot(1, 2, 1)
#     plt.scatter(starts[:, 0], starts[:, 1], s=5)
#     plt.title('start points')
#     plt.subplot(1, 2, 2)
#     plt.scatter(ends[:, 0], ends[:, 1], s=5)
#     plt.title('end points')
#     plt.show()

if __name__ == '__main__':
    read_raw_data()
    #completeTrajectories()
    #computeFeas()
    #generate_normal_behavior_sequence()


