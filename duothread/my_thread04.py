'''
 如果使用run 将会作为普通方法执行
'''


import  threading




#定义作为执行函数的

def action(max):
    for i in range(max):
        print(threading.current_thread().name  + " " + str(i))



for i in range(100):

    print(threading.current_thread().name + " " + str(i))


    if i == 20:
        threading.Thread(target=action,args=(100,)).run()
        threading.Thread(target=action,args=(100,)).run()


