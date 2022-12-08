import re

def getMultipleChoice(l):
    if len(l) == 0:
        print("Error")
        return
    if len(l) == 1:
        print("selected: " + l[0])
        return l[0]
    print( "\n".join ([str(i) + ") " + choice for i,choice in enumerate(l)]))
    input_ = input("choice (index or partial string): ")
    if re.findall("^\d+$",input_):
        if int(input_) in range(len(l)):
            return getMultipleChoice([l[int(input_)]])
    else:
        new_out = []
        for choice in l:
            if input_.lower() in choice.lower():
                new_out.append(choice)
        return getMultipleChoice(new_out)

