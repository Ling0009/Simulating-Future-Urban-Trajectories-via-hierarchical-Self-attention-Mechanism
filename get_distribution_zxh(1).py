import math
import pickle

import numpy as np
import scipy.stats

#from utils import merge_sets, cell_to_xynum

def cell_to_xynum(trajs, celly):
    return [[[int(cell / celly), int(cell) % celly] for cell in traj] for traj in trajs]

cellx = 250
celly = 250
max_locs = 62500
time_slice = 30 * 60  # seconds
time_step = math.ceil(24 * 60 * 60 / time_slice)  # 24小时对应的时间片数

set_num = 1
date_list = ['1001_1010', ]


def get_distances(trajs):
    seq_len = len(trajs)
    distances = np.zeros(shape=(seq_len), dtype=float)
    for j in range(seq_len):
        traj = trajs[j]
        for i in range(len(traj) - 1):
            dx = traj[i][0] - traj[i + 1][0]
            dy = traj[i][1] - traj[i + 1][1]
            distances[j] += dx ** 2 + dy ** 2 + 1e-4
    distances[distances > 100] = 99.9
    return distances


def get_visits(trajs, max_locs):
    """
    get probability distribution of visiting all locations
    :param trajs:
    :return:
    """
    visits = np.zeros(shape=(max_locs), dtype=float)
    for traj in trajs:
        for t in traj:
            visits[int(t)] += 1
    visits = visits
    return visits


# def get_durations(trajs):
#     d = []
#     for traj in trajs:
#         num = 1
#         for i, lc in enumerate(traj[1:]):
#             if lc == traj[i]:
#                 num += 1
#             else:
#                 d.append(num)
#                 num = 1
#     return np.array(d) / 48

def get_durations(trajs):
    d = []
    for traj in trajs:
        num = 1
        for i, lc in enumerate(traj[1:]):
            if lc == traj[i]:
                num += 1
            else:
                d.append(num)
                num = 1

    d = np.array(d)
    d[d > 100] = 99.9
    d[d < 0.1] = 0.1
    return d


def get_gradius(trajs):
    """
    get the std of the distances of all points away from center as `gyration radius`
    :param trajs:
    :return:
    """
    gradius = []
    for traj in trajs:
        seq_len = len(traj)
        xs = np.array([t[0] for t in traj])
        ys = np.array([t[1] for t in traj])
        xcenter, ycenter = np.mean(xs), np.mean(ys)
        dxs = xs - xcenter
        dys = ys - ycenter
        rad = [dxs[i] ** 2 + dys[i] ** 2 + 1e-4 for i in range(seq_len)]
        rad = np.mean(np.array(rad, dtype=float))
        gradius.append(rad)
    gradius = np.array(gradius, dtype=float)
    gradius[gradius > 100] = 99.9
    return gradius


def filter_zero(arr):
    """
    remove zero values from an array
    :param arr: np.array, input array
    :return: np.array, output array
    """
    arr = np.array(arr)
    filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
    return filtered_arr


def get_js_divergence(p1, p2):
    """
    calculate the Jensen-Shanon Divergence of two probability distributions
    :param p1:
    :param p2:
    :return:
    """
    # normalize
    p1 = p1 / (p1.sum() + 1e-14)
    p2 = p2 / (p2.sum() + 1e-14)
    m = (p1 + p2) / 2
    js = 0.5 * scipy.stats.entropy(p1, m) + \
         0.5 * scipy.stats.entropy(p2, m)
    return js


def arr_to_distribution(arr, min, max, bins):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param min: float, minimum of converted value
    :param max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """
    distribution, base = np.histogram(
        arr, np.arange(
            min, max + 1, int(
                max - min) / bins))
    return distribution, base[:-1]


def norm_arr_to_distribution(arr, bins=100):
    """
    normalize an array and convert it to distribution
    :param arr: np.array, input array
    :param bins: int, number of bins in [0, 1]
    :return: np.array, np.array
    """
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = filter_zero(arr)
    distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
    return distribution, base[:-1]


def log_arr_to_distribution(arr, min=-30., bins=100):
    """
    calculate the logarithmic value of an array and convert it to a distribution
    :param arr: np.array, input array
    :param bins: int, number of bins between min and max
    :return: np.array,
    """
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = filter_zero(arr)
    arr = np.log(arr)
    distribution, base = np.histogram(arr, np.arange(min, 0., 1. / bins))
    ret_dist, ret_base = [], []
    for i in range(bins):
        if int(distribution[i]) == 0:
            continue
        else:
            ret_dist.append(distribution[i])
            ret_base.append(base[i])
    return np.array(ret_dist), np.array(ret_base)


def get_score(true_trajs, pred_trajs):
    """
    :param true_trajs: 测试集的轨迹，格式示例[[1, 2, 3], [4, 5, 6]]，其中1 2 3为一条轨迹，1为第一个点的网格编号
    :param pred_trajs: baseline生成的轨迹，格式同上
    :return: 打分，分别是js_dict中各个得分的平均值，和各个得分的详细数据
    """
    # if validating:
    #     fname = 'validating'
    # else:
    #     fname = 'testing'
    # validating_distribution = pickle.load(open(f'./data/{fname}_distribution_dict', 'rb'))
    js_dict = {'distance': 0.0, 'visits_per_loc': 0.0, 'starter': 0.0, 'ender': 0.0,
               'loc_visited': 0.0, 'duration': 0.0, 'radius': 0.0}

    pred_trajs = get_slice_distribution(pred_trajs)
    true_trajs = get_slice_distribution(true_trajs)


    for key in js_dict.keys():
        js_dict[key] += get_js_divergence(pred_trajs[key][0],true_trajs[key][0])

    total_js = np.array(list(js_dict.values())).mean()

    pickle.dump(total_js, open('./data/total_js', 'wb+'))
    pickle.dump(js_dict, open('./data/js_dict', 'wb+'))

    return total_js, js_dict




def get_slice_distribution(slice):
    distribution_dict = {}
    xy_nums, cells = cell_to_xynum(slice, celly), slice
    # 单条轨迹的长度分布
    distribution_dict['distance'] = arr_to_distribution(get_distances(xy_nums),
                                                        min=0,
                                                        max=100,
                                                        bins=100)
    # 每个网格访问频率的分布
    distribution_dict['visits_per_loc'] = arr_to_distribution(get_visits(cells, max_locs),
                                                              min=0,
                                                              max=100,
                                                              bins=100)
    # 具体每个网格的访问频率
    distribution_dict['loc_visited'] = (get_visits(cells, max_locs), np.arange(0, max_locs+1, 1))
    starter = [x[0:1] for x in cells]
    ender = [x[-1:] for x in cells]
    distribution_dict['starter'] = (get_visits(starter, max_locs), np.arange(0, max_locs + 1, 1))
    distribution_dict['ender'] = (get_visits(ender, max_locs), np.arange(0, max_locs + 1, 1))
    # 每两个网格间的时间
    distribution_dict['duration'] = arr_to_distribution(get_durations(cells),
                                                        min=0,
                                                        max=100,
                                                        bins=100)
    # 一条轨迹的离散度
    distribution_dict['radius'] = arr_to_distribution(get_gradius(xy_nums),
                                                      min=0,
                                                      max=100,
                                                      bins=100)

    return distribution_dict


if __name__ == "__main__":

    # true_trajs, pred_trajs看函数定义的参数介绍
    Testhmm = pickle.load(open('./data/Testhmm', 'rb'))
    fdata = pickle.load(open('./data/fdata', 'rb'))
    mean_score, detail_score = get_score(Testhmm, fdata)
