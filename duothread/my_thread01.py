'''


采用threadpool 执行任务，可以很大程度上节约时间

'''


# pool = ThreadPool(poolsize)
# requests = makeRequests(some_callable, list_of_args, callback)
# [pool.putRequest(req) for req in requests]
# pool.wait()


import time

import threadpool


def sayHello(name):
    print('hello', name)
    time.sleep(2)


namelist = ['xiaozi', 'a', 'b', 'd']
start = time.time()
for i in range(len(namelist)):
    sayHello(namelist[i])

time_end = time.time()
print(time_end - start)

# 多线程跑
start_time = time.time()
pool = threadpool.ThreadPool(10)

requests = threadpool.makeRequests(sayHello, namelist)
[pool.putRequest(req) for req in requests]
pool.wait()

pool.wait()
end_time_thread = time.time()

print(end_time_thread - start_time)  # 2.0036404132843018

