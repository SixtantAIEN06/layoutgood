import recognize_faces_image
import logging
import numpy as np
a=recognize_faces_image.readPara("encoding/encoding_all_nj1_300p.pickle",'testSet/photo/ford001.jpg','hog',0.45)
print(a.keys())


# @profile
# def my_func():
#     import gc
#     a=5
#     con=True
#     while con:
#         try:
#             if a>0:
#                 b=0
#             else:
#                 b=1
#             c=5/b
#             print(c)
#             d=5
#             con2=True
#             while con2:
#                 try:
#                     z=[1]*100
#                     print(hex(id(z)))
#                     if d>0:
#                         e=0
#                     else:
#                         e=1
#                     f=5/e
#                     print(f)
#                     con2=False
#                 except ZeroDivisionError as e:
#                     d-=1
#                     print(f'In {e} happen , now d = {d}')
#                     del z
#                     gc.collect()
#             con=False
#         except ZeroDivisionError as e:
#             a-=1
#             print(f'Out {e} happen , now a = {a}')
# if __name__=='__main__':
#     my_func()
#----------------------------------------------------------------------------
# import gc
# @profile
# def my_func():
#     a=[1]*100000
#     b=[20]*1000000
#     print(hex(id(a)))
#     print(hex(id([1]*100000)))
#     print(hex(id(b)))
#     print(hex(id([20]*1000000)))
#     del b
#     gc.collect()
#     print(hex(id(a)))
#     print(hex(id([1]*100000)))
#     # print(hex(id(b)))
#     print(hex(id([20]*
# if __name__=='__main__':
#     my_func()
#------------------------------------------------------------------------------

# print('out of my_func')
# def power2(x):
#     x=x**2
#     return x
# def my_func():
#     a=5
#     b=power2(a)
#     print(b)
# if __name__=='__main__':
#     my_func()
#------------------------------------------------------------------------------

# print('out of class test')
# class test():
#     def __init__(self):
#         pass
#     def power2(self,x):
#         x=x**2
#         return x
#     def my_func(self):
#         a=5
#         b=self.power2(a)
#         print(b)
# if __name__=='__main__':
#     A=test()
#     A.my_func()
#--------------------------------------------------------------------------------------------------
# try:
#     a=5/0
#     print(a)
# except Exception as e:
#     print(type(e),type(e).__name__,e.__class__.__name__,e.__class__.__qualname__)
#-------------------------------------------------------------------------------------------------
# cond=True
# a=0
# while cond :
#     for _ in range(10):
#         print(_)
#         if _>5:
#             print(f'{_}>5')
#             break
#         elif _>=8:
#             cond=False
#-------------------------------------------------------------------------------------------------
# cond=True
# a=0
# li1=[2,4,5]
# while cond :
#     for _ in range(10):
#         if _ in li1:
#             continue
#         elif _>=8:
#             cond=False
#         print(_)
#-------------------------------------------------------------------------------------------------
# l1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
# l2=[0,1,2,3,4,5,6,7,8,9]
# bs=5
# last_batch=(bs*(len(l1)//bs-1))
# cond=True
# while cond:
#     for _ in range(0,len(l1),bs):
#         print(_)
#         print(l1[_:_+bs])
#         # if _==last_batch and len(l1)>last_batch:
#         #     for _ in range(last_batch+1,len(l1)):
#         #         print(_)
#         #         print('l1 : ',l1[_])
#     cond=False
#-------------------------------------------------------------------------------
# it=5
# while True :
#     it-=1
#     try:
#         a=1/it
#         print(a)
#     except:
#         print("it should be finised")
#         break
##------------------------------------------------------------------------------
# import sys, os

# try:
#     raise NotImplementedError("No error")
# except Exception as e:
#     exc_type, exc_obj, exc_tb = sys.exc_info()
#     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#     print(exc_type, fname, exc_tb.tb_lineno)
#--------------------------------------------------------------------------------------
# import os
# import subprocess as sp
# import logging
# import re

# encoding_file_list=sorted(os.listdir(os.path.dirname(os.path.abspath(__file__))+'/encoding'))
# r=re.compile("\w+[^all]\_nj\w+\.pickle")
# li1=sorted(list(filter(r.match,encoding_file_list)))
# # for _ in encoding_file_list:
# #     re.findall("\w+\_nj\w+\.pickle",)
# print(li1)






         