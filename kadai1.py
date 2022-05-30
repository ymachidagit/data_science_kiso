def method1():
    sum=1+2+3+4+5+6+7+8+9+10

    return sum

def method2():
    sum=0
    
    for i in range(10):
        sum+=(i+1)

    return sum

def method3():
    sum=0
    i=0

    while i<10:
        sum+=(i+1)
        i=i+1

    return sum

def method4():
    sum=0
    num_list=[1,2,3,4,5,6,7,8,9,10]

    for i in num_list:
        sum+=i
    
    return sum

print(f'方法1：{method1()}\n方法2：{method2()}\n方法3：{method3()}\n方法4：{method4()}')