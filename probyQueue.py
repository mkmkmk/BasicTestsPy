
import queue
import collections
from collections import deque


q = queue.Queue()
for i in range(5):
    q.put(i)

while not q.empty():
    print(q.get())


  
q = collections.deque()         
for i in range(5):
    q.append("{*}")
print(", ".join(q))
      
q = collections.deque()         
for i in range(5):
    q.append(i)

deque(range(5))


print(", ".join(map(lambda x: str(x), q)))
print(", ".join([str(i) for i in q]))
print(", ".join(str(i) for i in q))

            
print(str(4))

print(q[0], q[-1])

print(all(el >= 0 for el in q))


print(", ".join(map(lambda x: str(x), q)))
q.popleft()
print(", ".join(map(lambda x: str(x), q)))

#----------------------------------------------

for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)

for v in ['tic', 'tac', 'toe']:
    print(v)

#----------------------------------------------

import numpy as np
q = np.array(range(5)) * .333333    
print(", ".join("{:.2f}".format(i) for i in q))

#----------------------------------------------    
sum([x for x in range(10)])
sum(x for x in range(10))

    
