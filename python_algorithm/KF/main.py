import numpy as np

def test1():
    m1 = []
    m2 = []
    a1 = np.array([1, 2])
    a2 = np.array([3, 4])
    a3 = np.array([[1, 2], [3, 4]])
    a4 = np.array([[5, 6], [7, 8]])

    a5 = np.vstack((a1, a2))
    a6 = np.stack((a3, a4), axis=1)
    m1.append(a1)
    m2.append(a1.tolist())
    m2.append(a2.tolist())
    print(a5)
    print(a6)
    print(m1)
    print(m2)
    print(a3[:,np.newaxis].shape)

def test2():
    a1 = np.array([1, 2, 3, 4])
    a2 = np.array([5, 6, 7, 8])
    boxes = [a1, a2]

    box_num = len(boxes)
    frame_num = 5
    l = []
    for i in range(frame_num):
        for j in range(box_num):
            if i == 0:
                l.append([boxes[j]])
            else:
                l_b = l[j]
                l_b.append(boxes[j])
                l[j] = l_b

    print(l)

def test3():
    a1 = np.array([1, 2, 3])
    a2 = np.array([1, 2, 3])
    l1 = [a1, a2]
    a3 = np.array(l1)
    print(a3)

if __name__ == '__main__':
    # test1()
    # test2()
    test3()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
