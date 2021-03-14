import numpy as np
import pandas as pd
import heapq as hq
import sys
import copy
import time


# global variable
upperbound = sys.maxsize
update_upperbound_count = 0
best_permutations = []
job_length = 0
debug_list = []


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

# issue : srpt_version1還需要再改，因為估出來的時間比srpt_version2還少，可能有一兩個case沒考慮到


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
    flag = 0  # 如果走到最後一個i，但是沒有進去arrival_time<=current_time，就會直接start_index = job_amount - 1，然後把它全部做完，出事!!
    for i in range(job_amount):
        job_name = job_info[0][i]
        arrival_time = job_info[1][i]
        process_time = job_info[2][i]
        start_index = i
        if arrival_time <= current_time:
            hq.heappush(process_queue, [process_time, [
                        job_name, arrival_time, process_time]])
        else:
            flag = 1
            break

    # 代表整個heap內的arrival time都小於current time，所以就直接處理完，若在這不處理完，進入下面的for loop，會發生最後一個job重複push到heap的bug
    # 因為start_index == job_amount，所以會再走一次for loop內的動作
    # 如果flag不為0，代表他有走到最後一個index，但是他的arrival_time > current_time，所以這裡的動作就不能做，因為我是假設整個heap內的抵達時間都要小於目前的時間！
    if job_amount - 1 == start_index and flag == 0:
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
    # refer to manting's implementation
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


def bfs(arr):
    # arr[0] : job list
    # arr[1] : job arrival time
    # arr[2] : job process time
    # arr[3] : completion time of each job
    # return lowerbound_count
    global upperbound, best_permutations, job_length, update_upperbound_count, debug_list
    queue = []
    arr = copy.deepcopy(arr)
    total_job_list = arr[0]
    job_length = len(total_job_list)
    lowerbound_count = 0  # when lowerbound is calculate, count++

    # 產生初始資料進去heap
    for i in total_job_list:
        fixed_job = [x for x in total_job_list if x == i]
        temp_fixed_job = build_now_data(arr, fixed_job)
        unfixed_job = [x for x in total_job_list if x != i]
        temp_unfixed_job = build_now_data(arr, unfixed_job)
        fixed_time, last_job_finish_time = calculate_fixed_time(temp_fixed_job)
        srpt_time = srpt_version2(
            temp_unfixed_job, last_job_finish_time)
        lowerbound_now = fixed_time + srpt_time
        lowerbound_count += 1
        # print('lowerbound_count', lowerbound_count)
        # print('fixed_time', fixed_time, 'fixed_job', fixed_job)
        # print('srpt_time', srpt_time,
        #       'unfixed_job', unfixed_job)
        # debug
        # debug_list.append(['lowerbound_count ', str(lowerbound_count), '\nfixed_time ', str(fixed_time), ' fixed_job ', " ".join('%s' % id for id in fixed_job),
        #                    '\nsrpt_time ', str(srpt_time), ' unfixed_job ', " ".join('%s' % id for id in unfixed_job), '\n'])
        hq.heappush(queue, [lowerbound_now, fixed_job])

    current_job_process_time = queue[0][0]
    # 執行bfs
    while (current_job_process_time <= upperbound):
        fixed_job = hq.heappop(queue)[1]
        for i in total_job_list:
            if i not in fixed_job:
                walked_job = fixed_job + [i]
                remain_job = [x for x in total_job_list if x not in walked_job]
                temp_fixed_job = build_now_data(arr, walked_job)
                temp_unfixed_job = build_now_data(arr, remain_job)
                fixed_time, last_job_finish_time = calculate_fixed_time(
                    temp_fixed_job)
                srpt_time = srpt_version2(
                    temp_unfixed_job, last_job_finish_time)
                lowerbound_now = fixed_time + srpt_time
                if len(walked_job) == job_length:
                    lowerbound_count += 1  # record visited node
                    # print('lowerbound_count', lowerbound_count)
                    # print('fixed_time', fixed_time, 'fixed_job', walked_job)
                    # print('srpt_time', srpt_time,
                    #       'unfixed_job', remain_job)
                    # print('fixeded current_job_process_time', current_job_process_time,
                    #       'lowerbound_now', lowerbound_now, 'upperbound', upperbound)
                    # debug
                    # debug_list.append(['lowerbound_count ', str(lowerbound_count), '\nfixed_time ', str(fixed_time), ' fixed_job ', " ".join('%s' % id for id in walked_job),
                    #                    '\nsrpt_time ', str(srpt_time), ' unfixed_job ', " ".join('%s' % id for id in remain_job), '\nfixeded current_job_process_time ', str(
                    #                        current_job_process_time),
                    #                    ' lowerbound_now ', str(lowerbound_now), ' upperbound ', str(upperbound), '\n'])
                    if lowerbound_now < upperbound:
                        update_upperbound_count += 1
                        upperbound = lowerbound_now
                        best_permutations.clear()
                        best_permutations.append(walked_job)
                    elif lowerbound_now == upperbound:
                        best_permutations.append(walked_job)
                    # else: # 其他的就是被bounded job
                else:
                    if lowerbound_now <= upperbound:
                        lowerbound_count += 1  # record visited node
                        # print('lowerbound_count', lowerbound_count)
                        # print('fixed_time', fixed_time,
                        #       'fixed_job', walked_job)
                        # print('srpt_time', srpt_time,
                        #       'unfixed_job', remain_job)
                        # print('fixeded current_job_process_time', current_job_process_time,
                        #       'lowerbound_now', lowerbound_now, 'upperbound', upperbound)
                        # debug
                        # debug_list.append(['lowerbound_count ', str(lowerbound_count), '\nfixed_time ', str(fixed_time), ' fixed_job ', " ".join('%s' % id for id in walked_job),
                        #                    '\nsrpt_time ', str(srpt_time), ' unfixed_job ', " ".join('%s' % id for id in remain_job), '\nfixeded current_job_process_time ', str(
                        #                        current_job_process_time),
                        #                    ' lowerbound_now ', str(lowerbound_now), ' upperbound ', str(upperbound), '\n'])
                        hq.heappush(queue, [lowerbound_now, walked_job])
        current_job_process_time = queue[0][0]
        # print('updated current_job_process_time', current_job_process_time,
        #       'lowerbound_now', lowerbound_now, 'upperbound', upperbound)
        # debug
        # debug_list.append(['\nupdated current_job_process_time ', str(current_job_process_time),
        #                    ' lowerbound_now ', str(lowerbound_now), ' upperbound ', str(upperbound), '\n'])
    return lowerbound_count


def six_test_data():
    global job_length
    job_list = [1, 2, 3, 4, 5, 6]
    Rj = [0, 2, 2, 6, 7, 9]
    Pj = [6, 2, 3, 2, 5, 2]
    # job_list = [1, 2, 3, 4]
    # Rj = [0, 2, 2, 6]
    # Pj = [6, 2, 3, 2]
    n = len(job_list)
    # data[3] : completion time of each job
    completed_time = [0] * n
    data = []
    data.append(job_list)
    data.append(Rj)
    data.append(Pj)
    data.append(completed_time)
    data = np.array(data)
    job_length = len(data[0])

    # bfs
    start = time.time()
    visited_node_amount = bfs(data)
    for permutation in best_permutations:
        print('best_permutation : ', permutation)
    print('objective_value : ', upperbound)
    print('visited_node_amount : ', visited_node_amount)
    end = time.time()
    print("elapsed run time is {} seconds".format(end - start))


def all_test_data_from_hw1(k):
    # k : want to test how many tests, 0 <= k <= 100
    global upperbound, best_permutations, job_length, update_upperbound_count
    # header：指定作為列名的行，預設0，即取第一行，資料為列名行以下的資料；若資料不含列名，則設定 header = None；
    df = pd.read_excel('test instance.xlsx', header=None)
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

    # bfs_branch and bound
    test_case = job_total_info[:, :k]
    job_length = len(test_case[0])
    start = time.time()
    visited_node_amount = bfs(test_case)
    for permutation in best_permutations:
        print('best_permutation : ', permutation)
    print('objective_value : ', upperbound)
    print('visited_node_amount : ', visited_node_amount)
    print('update_upperbound_count', update_upperbound_count)
    end = time.time()
    print("elapsed run time is {} seconds".format(end - start))

    # debug
    # f = open('srpt_version1.txt', 'w')
    # for each in debug_list:
    #     f.writelines(each)
    # f.close()

    # initial global variable
    job_length = 0
    upperbound = sys.maxsize
    best_permutations = []
    update_upperbound_count = 0


def main():

    # six_test_data()
    for i in range(6, 16, 1):
        print('==========the {} experiment=========='.format(i))
        all_test_data_from_hw1(i)
    # all_test_data_from_hw1(15)


main()
