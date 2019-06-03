'''

线程join
'''


import threading


def action(max):

    for i in range(max):
        print(threading.current_thread().name + ' ' +str(i))



#启动子线程

threading.Thread(target=action,args=(100,),name='新线程').start()



for i in range(100):

    if i == 20:
        jt = threading.Thread(target=action, args=(100,), name="joined thread")
        jt.start()
        # 主线程调用了jt线程的join()方法，主线程
        # 必须等jt执行结束才会向下执行

        jt.join()

    print(threading.current_thread().name + " " + str(i))



