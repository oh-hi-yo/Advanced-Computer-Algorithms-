import numpy as np
import pandas as pd
import heapq as hq
import sys
import copy
import time


# global variable
upperbound = sys.maxsize
best_permutations = []
job_length = 0
lowerbound_count = 0  # when lowerbound is calculate, count++
total_job = []
update_upperbound_count = 0


def swap_cols(arr, frm, to):
    arr[:, [frm, to]] = arr[:, [to, frm]]


def calculate_fixed_time(arr):
    # arr[0] : job list
    # arr[1] : job arrival time
    # arr[2] : job process time
    # arr[3] : completion time of each job

    current_time = 0
    job_info = copy.deepcopy(arr)
    job_amount = len(job_info[0])

    for i in range(job_amount):
        job_name = job_info[0][i]
        arrival_time = job_info[1][i]
        process_time = job_info[2][i]
        if(arrival_time > current_time):  # 目前時間尚未到job的到達時間
            current_time += (arrival_time - current_time)
            current_time += process_time
        else:
            current_time += process_time
        job_info[3][i] = current_time
    return sum(job_info[3]), job_info[3][job_amount - 1]


def srpt_version1(arr, last_job_finish_time):
    # arr[0] : job list
    # arr[1] : job arrival time
    # arr[2] : job process time
    # arr[3] : completion time of each job
    # last_job_finish_time : the last job finish time
    # [process time, [job name, arrival time, process time]] is the data structure in the heap

    current_time = last_job_finish_time
    process_queue = []  # store the unfinished job
    # copy v.s. deepcopy, deepcopy can address memory issue in shallow copy, refer to : https://ithelp.ithome.com.tw/articles/10221255
    job_info = copy.deepcopy(arr)
    job_amount = len(job_info[0])

    # 先把arrival time <= current_time的push進去heap queue
    start_index = 0  # 紀錄目前執行到哪個index
    for i in range(job_amount):
        job_name = job_info[0][i]
        arrival_time = job_info[1][i]
        process_time = job_info[2][i]
        start_index = i
        if arrival_time <= current_time:
            hq.heappush(process_queue, [process_time, [
                        job_name, arrival_time, process_time]])
        else:
            break

    # 代表整個heap內的arrival time都小於current time，所以就直接處理完，若在這不處理完，進入下面的for loop，會發生最後一個job重複push到heap的bug
    # 因為start_index == job_amount，所以會再走一次for loop內的動作
    if job_amount - 1 == start_index:
        while len(process_queue) != 0:
            pop_job = hq.heappop(process_queue)
            min_time = pop_job[0]
            current_job_name = pop_job[1][0]
            current_time += min_time
            job_index = np.where(job_info[0] == current_job_name)  # 返回索引
            job_index = job_index[0][0]
            job_info[3][job_index] = current_time

        return sum(job_info[3])

    # 做剩下的job
    for i in range(start_index, job_amount):
        job_name = job_info[0][i]
        arrival_time = job_info[1][i]
        process_time = job_info[2][i]
        if arrival_time <= current_time:
            hq.heappush(process_queue, [process_time, [
                        job_name, arrival_time, process_time]])
            if i < (job_amount - 1):  # 最後一個job就沒有可以用的時間了，因為沒有下一個job，但最後一個job也一定會被push進去heap
                next_job_arrival_time = job_info[1][i + 1]
                can_use_time = next_job_arrival_time - arrival_time
                try:
                    current_job_process_time = process_queue[0][0]
                    while can_use_time >= current_job_process_time:
                        # 可以執行目前剩餘時間最小的process, must finish this job, 這邊才會紀錄到完工時間
                        if len(process_queue) == 0:  # 若process_queue內已經沒東西了，就不用做下去了
                            break
                        pop_job = hq.heappop(process_queue)
                        min_time = current_job_process_time
                        current_job_name = pop_job[1][0]
                        current_time += min_time
                        can_use_time -= min_time
                        job_index = np.where(
                            job_info[0] == current_job_name)  # 返回索引
                        job_index = job_index[0][0]
                        job_info[3][job_index] = current_time
                        # update current_job_process_time
                        current_job_process_time = process_queue[0][0]

                    if len(process_queue) != 0:
                        # address process_time in heap
                        process_queue[0][0] -= can_use_time
                        # address process_time in heap
                        process_queue[0][1][2] -= can_use_time
                        current_time += can_use_time
                    else:
                        current_time += can_use_time
                except:
                    pass
        else:
            # arrival_time > current_time
            can_use_time = arrival_time - current_time
            try:
                current_job_process_time = process_queue[0][0]
                while can_use_time >= current_job_process_time:
                    # 可以執行目前剩餘時間最小的process, must finish this job, 這邊才會紀錄到完工時間
                    if len(process_queue) == 0:  # 若process_queue內已經沒東西了，就不用做下去了
                        break
                    pop_job = hq.heappop(process_queue)
                    min_time = current_job_process_time
                    current_job_name = pop_job[1][0]
                    can_use_time -= min_time
                    current_time += min_time
                    job_index = np.where(
                        job_info[0] == current_job_name)  # 返回索引
                    job_index = job_index[0][0]
                    job_info[3][job_index] = current_time
                    # update current_job_process_time
                    current_job_process_time = process_queue[0][0]

                if len(process_queue) != 0:
                    # address process_time in heap
                    process_queue[0][0] -= can_use_time
                    # address process_time in heap
                    process_queue[0][1][2] -= can_use_time
                    current_time += can_use_time
                else:
                    current_time += can_use_time
            except:
                pass
            # 因為已經有等時間了，所以最後還是得把目前的job information加進去heap
            hq.heappush(process_queue, [process_time, [
                        job_name, arrival_time, process_time]])

    while len(process_queue) != 0:  # 若上面都做完了，但heap內還有東西
        pop_job = hq.heappop(process_queue)
        min_time = pop_job[0]
        current_job_name = pop_job[1][0]
        current_time += min_time
        job_index = np.where(job_info[0] == current_job_name)  # 返回索引
        job_index = job_index[0][0]
        job_info[3][job_index] = current_time
    return sum(job_info[3])


def srpt_version2(arr, last_job_finish_time):
    current_time = last_job_finish_time
    completed_job_amount = 0
    process_queue = []  # store the unfinished job
    job_info = copy.deepcopy(arr)
    job_amount = len(job_info[0])
    objective_value = 0
    # job_list = arr[0] # store job name

    i = 0  # record job index
    try:
        job_name = job_info[0][i]
        arrival_time = job_info[1][i]
        process_time = job_info[2][i]
    except:
        pass
    while completed_job_amount != job_amount:
        while i < job_amount and arrival_time <= current_time:
            hq.heappush(process_queue, [process_time, [
                        job_name, arrival_time, process_time]])
            i += 1  # increase job index
            if i == job_amount:
                break
            # print('i', i)
            job_name = job_info[0][i]
            arrival_time = job_info[1][i]
            process_time = job_info[2][i]
        if len(process_queue) != 0 and process_queue[0][0] == 0:
            hq.heappop(process_queue)
            completed_job_amount += 1
            objective_value += current_time
        if len(process_queue) != 0:
            # address process_time in heap
            process_queue[0][0] -= 1
            # address process_time in heap
            process_queue[0][1][2] -= 1

        current_time += 1

    return objective_value


def build_now_data(arr, job_list):
    # arr : 本來的資料
    # job_list : job_list is a 1D array, store job
    arr = copy.deepcopy(arr)
    length = len(job_list)
    temp = np.zeros((4, length))
    for k in range(length):
        for j in range(len(arr[0])):
            if job_list[k] == arr[0, j]:
                temp[0, k] = arr[0, j]
                temp[1, k] = arr[1, j]
                temp[2, k] = arr[2, j]
    return temp


def dfs1(arr, l, r):
    # arr[0] : job list
    # arr[1] : job arrival time
    # arr[2] : job process time
    # arr[3] : completion time of each job

    global upperbound, best_permutations, job_length, lowerbound_count
    if (l == r):  # 排到底了
        fixed_job = arr[:, :l + 1]
        unfixed_job = arr[:, l + 1:]
        fixed_time, last_job_finish_time = calculate_fixed_time(fixed_job)
        srpt_time = srpt_version1(unfixed_job, last_job_finish_time)
        lowerbound_now = fixed_time + srpt_time
        lowerbound_count += 1
        if (upperbound > lowerbound_now):  # update Lower bound
            upperbound = lowerbound_now
            best_permutations.clear()
            best_permutations.append(np.copy(fixed_job[0]))
        elif(upperbound == lowerbound_now):
            best_permutations.append(np.copy(fixed_job[0]))
        # else: # 其他的就是被bounded job

        # print('lowerbound_lod', upperbound,
        #       'lowerbound_now', lowerbound_now)
        # print('now permutation : ', fixed_job[0])
    else:
        for i in range(l, r+1):
            swap_cols(arr, l, i)

            # explannation how to get fixed sequence and unfixed sequence
            # print('fixed seq\n', arr[:, :l + 1])
            # print('unfixed seq\n', arr[:, l + 1:])
            fixed_job = arr[:, :l + 1]
            unfixed_job = arr[:, l + 1:]
            fixed_time, last_job_finish_time = calculate_fixed_time(fixed_job)
            srpt_time = srpt_version1(
                unfixed_job, last_job_finish_time)
            # print('srpt_time ', srpt_time, 'srpt_version1_sequence', srpt_version1_sequence)
            lowerbound_now = fixed_time + srpt_time
            # lowerbound_count += 1
            if upperbound >= lowerbound_now:  # 可以繼續做下去
                lowerbound_count += 1
                dfs1(arr, l+1, r)
            # else: # 其他的就是被bounded job

            swap_cols(arr, l, i)


def dfs2(arr, all_seq, walked_seq):
    # 佑誠版本
    global job_length, total_job, upperbound, lowerbound_count, best_permutations, update_upperbound_count
    leftest = 0
    for i in all_seq:
        if leftest > 0:  # 只有每個分支的最左邊不需要取出walked_seq的最後一個job, 其他都要取出來，不然walked job會一直包含該分支最左邊的job
            walked_seq = walked_seq[:len(walked_seq) - 1] + [i]
        else:
            walked_seq = walked_seq + [i]

        leftest += 1
        remain_seq = [x for x in total_job if x not in walked_seq]

        fixed_job = build_now_data(arr, walked_seq)
        unfixed_job = build_now_data(arr, remain_seq)
        fixed_time, last_job_finish_time = calculate_fixed_time(fixed_job)
        srpt_time = srpt_version2(unfixed_job, last_job_finish_time)
        lowerbound_now = fixed_time + srpt_time
        # lowerbound_count += 1 # 這裡是每個點都掃看看，但是要看走過哪些點要去看有沒有符合條件，有符合條件才會走下去，才要記錄。
        # print('fixed_time', fixed_time, 'fixed_job', fixed_job[0])
        # print('srpt_time', srpt_time, 'unfixed_job', unfixed_job[0])
        # print('last_job_finish_time', last_job_finish_time)

        if lowerbound_now <= upperbound:
            # print('lowerbound_now', lowerbound_now,
            #       'non-leaf upperbound', upperbound)
            lowerbound_count += 1
            dfs2(arr, remain_seq, walked_seq)
        # else: # 其他的就是被bounded job

        if(len(walked_seq) == job_length):
            lowerbound_count += 1
            lowerbound_now = fixed_time
            if (upperbound > lowerbound_now):  # update upperbound
                update_upperbound_count += 1
                upperbound = lowerbound_now
                best_permutations.clear()
                best_permutations.append(walked_seq)
                # print('lowerbound_now', lowerbound_now,
                #       'leaf upperbound', upperbound)
            elif(upperbound == lowerbound_now):
                best_permutations.append(walked_seq)
            # else: # 其他的就是被bounded job

            # print('upperbound', upperbound,
            #       'lowerbound_now', lowerbound_now)
            # print('now permutation : ', walked_seq)


def six_test_data():
    global upperbound, best_permutations, job_length, total_job, lowerbound_count
    job_list = [1, 2, 3, 4, 5, 6]
    Rj = [0, 2, 2, 6, 7, 9]
    Pj = [6, 2, 3, 2, 5, 2]
    # job_list = [1, 2, 3, 4]
    # Rj = [0, 2, 2, 6]
    # Pj = [6, 2, 3, 2]
    n = len(job_list)
    # data[3] : completion time of each job
    completed_time = [0] * n
    total_job = job_list[:]

    data = []
    data.append(job_list)
    data.append(Rj)
    data.append(Pj)
    data.append(completed_time)
    data = np.array(data)

    # dfs
    start = time.time()
    job_length = len(data[0])
    dfs1(data, 0, job_length-1)
    # dfs2(data, data[0], [])
    for permutation in best_permutations:
        print('best_permutation : ', permutation)
    print('objective_value : ', upperbound)
    visited_node_amount = lowerbound_count
    print('visited_node_amount : ', visited_node_amount)
    end = time.time()
    print("elapsed run time is {} seconds".format(end - start))


def all_test_data_from_hw1(k):
    # k : want to test how many tests, 0 <= k <= 100
    global upperbound, best_permutations, job_length, total_job, lowerbound_count, update_upperbound_count
    # header：指定作為列名的行，預設0，即取第一行，資料為列名行以下的資料；若資料不含列名，則設定 header = None；
    df = pd.read_excel('test instance.xlsx', header=None)
    # print(df)
    process_time = np.array(df.iloc[0, 1:].tolist())
    arrival_time = np.array(df.iloc[1, 1:].tolist())
    complete_time = np.zeros((len(process_time)))
    # 自己造job name
    job_name_list = np.array(df.columns.copy())
    job_name_list = job_name_list[1:]  # job name is '1' to '100'
    job_total_info = []
    job_total_info.append(job_name_list)
    job_total_info.append(arrival_time)
    job_total_info.append(process_time)
    job_total_info.append(complete_time)
    job_total_info = np.array(job_total_info)

    # dfs_branch and bound
    start = time.time()
    test_case = job_total_info[:, :k]
    total_job = test_case[0]
    # print('test_case\n', test_case)
    job_length = len(test_case[0])
    # dfs1(test_case, 0, job_length-1)
    dfs2(test_case, test_case[0], [])
    for permutation in best_permutations:
        print('best_permutation : ', permutation)
    print('objective_value : ', upperbound)
    visited_node_amount = lowerbound_count
    print('visited_node_amount : ', visited_node_amount)
    print('update_upperbound_count', update_upperbound_count)
    end = time.time()
    print("elapsed run time is {} seconds".format(end - start))

    # initial global variable
    job_length = 0
    lowerbound_count = 0
    upperbound = sys.maxsize
    best_permutations = []
    update_upperbound_count = 0


def main():

    # six_test_data()
    for i in range(6, 16, 1):
        print('==========the {} experiment=========='.format(i))
        all_test_data_from_hw1(i)
    # all_test_data_from_hw1(20)


main()
