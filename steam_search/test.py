import sys
line = input()

stack = []
for i in line:
    if i !=']':
        stack.append(i)
    else:
        temp = ''
        num = ''
        while stack[-1] != '[':
            if stack[-1].isalpha():
                temp =  stack.pop() + temp
            elif stack[-1] == '|':
                stack.pop()
            elif stack[-1].isdigit():
                num = stack.pop() + num
        stack.pop()
        stack.append(int(num) * temp)
print('wjg'.join(stack))
